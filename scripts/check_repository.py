"""Check maintained documentation, CFF metadata and frozen inputs without network access."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def anchors(text):
    counts, result = {}, set()
    for line in re.sub(r'```.*?```', '', text, flags=re.S).splitlines():
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if heading:
            label = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', heading[1])
            slug = re.sub(r'[^\w\- ]', '', label.lower()).replace(' ', '-')
            count = counts.get(slug, 0)
            counts[slug] = count + 1
            result.add(slug + (f'-{count}' if count else ''))
    result.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    return result


def check_links():
    errors, count = [], 0
    docs = [ROOT / 'README.md', ROOT / 'README_BENCHMARK_SUPPLEMENT.md', ROOT / 'CONTRIBUTING.md']
    docs += list((ROOT / 'docs').glob('*.md'))
    docs += [ROOT / 'training/pretrain/README.md', ROOT / 'finetune/aptamer/script/README.md',
             ROOT / 'reproduction/milena_14b/expected/README.md',
             ROOT / 'notebooks/README.md', ROOT / 'reproduction/README.md']
    for path in docs:
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        links = re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)', text)
        links += re.findall(r'(?:href|src)=["\']([^"\']+)', text)
        for target in links:
            target = html.unescape(target.strip('<>'))
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            count += 1
            local = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not local.is_relative_to(ROOT) or not local.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing local target {target}')
            elif url.fragment and local.suffix == '.md' and unquote(url.fragment) not in anchors(local.read_text()):
                errors.append(f'{path.relative_to(ROOT)}: missing section {target}')
    if errors:
        raise ValueError('\n'.join(errors))
    return count


def main():
    import jsonschema
    import yaml

    citation = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
    schema = json.loads((ROOT / '.github/citation-file-format-1.2.0.schema.json').read_text())
    jsonschema.Draft7Validator(schema, format_checker=jsonschema.FormatChecker()).validate(citation)
    version = re.search(r'__version__\s*=\s*["\']([^"\']+)', (ROOT / 'eva/_version.py').read_text())[1]
    if citation.get('version') != version:
        raise ValueError('CITATION.cff version does not match the package version')
    links = check_links()
    # Validate the actual frozen assay and historical-source checksums.
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.reproduce_milena import validate_inputs
    from scripts.reproduce_historical_benchmark import audit
    manifest, records, labels, _, _ = validate_inputs()
    report, rows = audit()
    if len(records) != manifest['n'] or len(labels) != len(rows) or not report['arithmetic_match']:
        raise ValueError('Frozen benchmark input or arithmetic check failed')
    print(json.dumps({'status': 'PASS', 'local_links_checked': links,
                      'citation_schema': 'CFF 1.2.0', 'version': version,
                      'frozen_assay_rows': len(records), 'model_inference': False}, indent=2))


if __name__ == '__main__':
    main()
