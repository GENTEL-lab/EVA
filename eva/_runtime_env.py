"""Runtime self-healing for Triton's CUDA driver discovery.

Triton 3.1.0 compiles ``cuda_utils.so`` from
``triton/backends/nvidia/driver.c`` on first use, linking it with
``gcc main.c -shared -lcuda -L<dirs>``. The ``-lcuda`` lookup happens at
link time through the ldconfig cache and ``LIBRARY_PATH``; it does not
follow ``LD_LIBRARY_PATH``, which is what ``dlopen`` uses at run time.

Container images produced via ``docker commit`` of running GPU containers
contain 0-byte ``libcuda.so.*`` placeholders (the runtime driver was a bind
mount, which ``commit`` records as empty files). Docker with ``--gpus``
bind-mounts the real driver over them and masks the defect. Singularity/
Apptainer ``--nv`` instead injects the host driver into ``/.singularity.d/
libs`` and ``LD_LIBRARY_PATH``, leaving the placeholders visible: Triton
links against the empty file and every later import fails with
``ImportError: undefined symbol: cuModuleGetFunction``.

This module is a no-op when Triton's default discovery works (regular
Docker). Otherwise it locates the real driver library and exports
``TRITON_LIBCUDA_PATH`` -- which ``triton/backends/nvidia/driver.py ::
libcuda_dirs`` honours with the highest priority, ahead of the ldconfig
cache -- then purges mis-linked cached Triton helper libraries (their cache
key only depends on the driver.c source, so a bad artifact would otherwise
be reused forever).

It runs automatically on ``import eva`` (see ``eva/__init__.py``), which
also covers ``singularity exec/shell`` sessions where container
ENTRYPOINTs do not run. Shell entrypoints consume it via
``python -m eva._runtime_env --sh``. Set ``EVA_SKIP_RUNTIME_ENV=1`` to
disable.
"""

from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
import sys

# Directories Singularity/Apptainer --nv uses to inject host driver libraries.
_SINGULARITY_LIB_DIRS = ("/.singularity.d/libs",)

# Standard driver locations (Docker with nvidia-container-toolkit, or CUDA
# compat packages).
_STANDARD_LIBCUDA_PATHS = (
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/local/cuda/compat/libcuda.so.1",
)

# Environment variables this module may export.
_MANAGED_VARS = ("TRITON_LIBCUDA_PATH", "TRITON_CACHE_DIR")


def _log(message: str) -> None:
    print(f"[eva.runtime_env] {message}", file=sys.stderr)


def _uid() -> int:
    getuid = getattr(os, "getuid", None)
    return getuid() if callable(getuid) else 0


def _valid_libcuda(path: str) -> str | None:
    """Return the real file behind ``path`` if it is a usable libcuda library.

    Rejects dangling symlinks and the 0-byte placeholder files baked into
    ``docker commit``-based images.
    """
    try:
        if not os.path.lexists(path):
            return None
        real = os.path.realpath(path)
        if os.path.isfile(real) and os.path.getsize(real) > 0:
            return real
    except OSError:
        pass
    return None


def _default_discovery_ok() -> bool:
    """Replicate Triton 3.1.0's default libcuda discovery and test it.

    Triton resolves ``-lcuda`` through ``/sbin/ldconfig -p``; when that cache
    already points at a usable (non-empty) libcuda, no intervention is needed.
    """
    try:
        cache = subprocess.run(
            ["/sbin/ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in cache.splitlines():
        if "libcuda.so.1" in line and _valid_libcuda(line.split()[-1]):
            return True
    return False


def _find_real_libcuda() -> str | None:
    """Locate a real, non-empty libcuda shared library."""
    # 1) LD_LIBRARY_PATH (Singularity --nv injects the host driver here) and
    #    the Singularity/Apptainer default injection directories.
    search_dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
    search_dirs.extend(_SINGULARITY_LIB_DIRS)
    for directory in search_dirs:
        real = _valid_libcuda(os.path.join(directory, "libcuda.so.1"))
        if real:
            return real

    # 2) Standard driver locations (regular Docker runtime).
    for path in _STANDARD_LIBCUDA_PATHS:
        real = _valid_libcuda(path)
        if real:
            return real

    # 3) Ask the dynamic loader itself: dlopen consults LD_LIBRARY_PATH even
    #    when the ldconfig cache is stale or poisoned, so /proc/self/maps
    #    reveals whichever libcuda this process can actually load.
    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            for line in maps:
                parts = line.split()
                if not parts:
                    continue
                candidate = parts[-1]
                if "libcuda.so" not in candidate or not candidate.startswith("/"):
                    continue
                if candidate.endswith("(deleted)"):
                    candidate = candidate[: -len("(deleted)")].rstrip()
                real = _valid_libcuda(candidate)
                if real:
                    return real
    except OSError:
        return None
    return None


def _default_triton_cache() -> str:
    override = os.environ.get("TRITON_CACHE_DIR")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".triton", "cache")


def _purge_stale_triton_artifacts(cache_root: str) -> None:
    """Delete cached Triton helper libraries that may have been mis-linked.

    ``cuda_utils.so`` and ``__triton_launcher*.so`` are rebuilt from source
    in about a second on first use, so always purging them here is cheap.
    """
    if not os.path.isdir(cache_root):
        return
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        for name in filenames:
            if name == "cuda_utils.so" or name.startswith("__triton_launcher"):
                try:
                    os.remove(os.path.join(dirpath, name))
                except OSError:
                    pass


def _prepare_link_dir(link_dir: str, real: str) -> bool:
    """Create ``libcuda.so`` / ``libcuda.so.1`` symlinks to ``real``.

    Tolerates concurrent processes preparing the same directory (e.g. Jupyter
    spawning multiple workers).
    """
    try:
        os.makedirs(link_dir, exist_ok=True)
        for link_name in ("libcuda.so", "libcuda.so.1"):
            link_path = os.path.join(link_dir, link_name)
            try:
                if os.path.lexists(link_path):
                    if os.path.realpath(link_path) == real:
                        continue
                    os.remove(link_path)
                os.symlink(real, link_path)
            except OSError:
                # Another process may have created/removed it concurrently.
                pass
    except OSError:
        return False
    return _valid_libcuda(os.path.join(link_dir, "libcuda.so.1")) is not None


def apply() -> dict[str, str]:
    """Fix the Triton runtime environment when its default discovery is broken.

    Returns the environment variables exported by this call. Idempotent and
    silent when Triton already resolves a usable driver (regular Docker).
    """
    exported: dict[str, str] = {}

    if os.environ.get("EVA_SKIP_RUNTIME_ENV") == "1":
        return exported

    existing = os.environ.get("TRITON_LIBCUDA_PATH", "")
    if existing and _valid_libcuda(os.path.join(existing, "libcuda.so.1")):
        # Triton is already pointed at a working driver; nothing to do.
        return exported

    if _default_discovery_ok():
        # Triton's own ldconfig-based discovery works (regular Docker).
        return exported

    real = _find_real_libcuda()
    if real is None:
        _log(
            "WARNING: no usable libcuda.so.1 found; Triton GPU kernels will "
            "not load. Launch with --gpus all (Docker) or --nv (Singularity/"
            "Apptainer)."
        )
        return exported

    link_dir = os.environ.get(
        "EVA_TRITON_LIBCUDA_DIR", f"/tmp/eva-triton-libcuda-{_uid()}"
    )
    if not _prepare_link_dir(link_dir, real):
        _log(f"WARNING: could not prepare {link_dir}")
        return exported

    os.environ["TRITON_LIBCUDA_PATH"] = link_dir
    exported["TRITON_LIBCUDA_PATH"] = link_dir
    _log(f"TRITON_LIBCUDA_PATH={link_dir} -> {real}")

    # Triton needs a writable cache directory (HPC home directories are
    # sometimes read-only); fall back to /tmp when the default is not usable.
    cache_root = _default_triton_cache()
    try:
        os.makedirs(cache_root, exist_ok=True)
        writable = os.access(cache_root, os.W_OK)
    except OSError:
        writable = False
    if not writable:
        fallback = f"/tmp/triton-cache-{_uid()}"
        try:
            os.makedirs(fallback, exist_ok=True)
        except OSError as exc:
            _log(f"WARNING: could not create Triton cache fallback {fallback}: {exc}")
            return exported
        os.environ["TRITON_CACHE_DIR"] = fallback
        exported["TRITON_CACHE_DIR"] = fallback
        _log(f"TRITON_CACHE_DIR={fallback} (default cache is not writable)")

    _purge_stale_triton_artifacts(cache_root)
    return exported


def _main() -> int:
    """CLI entry: ``python -m eva._runtime_env [--sh]``.

    With ``--sh``, prints the effective Triton environment as shell ``export``
    statements for container entrypoints to eval. Exit code is 0 when Triton
    ends up with a usable driver path (directly or via default discovery).
    """
    apply()
    if "--sh" in sys.argv[1:]:
        for key in _MANAGED_VARS:
            value = os.environ.get(key)
            if value:
                print(f"export {key}={shlex.quote(value)}")
    healthy = bool(os.environ.get("TRITON_LIBCUDA_PATH")) or _default_discovery_ok()
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(_main())
