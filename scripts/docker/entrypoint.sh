#!/bin/bash
# EVA container entrypoint: export the Triton CUDA driver environment before
# handing off to the user command. Detection logic lives in
# eva/_runtime_env.py (also runs on every `import eva`, which covers
# `singularity exec/shell` sessions where this entrypoint is not executed).

EVA_QUIET="${EVA_QUIET:-0}"
log() { [[ "$EVA_QUIET" == "1" ]] || echo "[eva-entrypoint] $*" >&2; }

py_bin="$(command -v python || command -v python3 || true)"
if [[ -n "$py_bin" ]]; then
    if exports="$("$py_bin" -m eva._runtime_env --sh 2>/dev/null)"; then
        if [[ -n "$exports" ]]; then
            eval "$exports"
            log "$exports"
        fi
    else
        log "WARNING: no usable libcuda detected; Triton GPU kernels may fail."
        log "Singularity/Apptainer users must launch the container with --nv."
    fi
else
    log "WARNING: python not found; skipping Triton libcuda self-healing."
fi

# Hand off to the NVIDIA entrypoint when running under Docker; it execs "$@"
# itself.
if [[ -x /opt/nvidia/nvidia_entrypoint.sh ]]; then
    exec /opt/nvidia/nvidia_entrypoint.sh "$@"
fi
exec "$@"
