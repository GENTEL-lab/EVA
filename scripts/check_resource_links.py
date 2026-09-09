"""Report public resource accessibility separately from offline regression checks.

Exit 0 means the access report was written, not that every URL is available.
This tool never downloads weight bodies or replaces a historical revision.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def probe(entry):
    item = dict(entry)
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'EVA-resource-check/1.0',
                                                            'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            item.update(http_status=response.status, resolved_url=response.url, access='AVAILABLE')
            if entry.get('api_revision'):
                data = json.load(response)
                item['observed_revision'] = data.get('sha')
                if entry.get('expected_revision') and data.get('sha') != entry['expected_revision']:
                    item['access'] = 'REVISION_MISMATCH'
    except urllib.error.HTTPError as exc:
        item.update(http_status=exc.code, access='UNAVAILABLE')
    except (OSError, ValueError) as exc:
        item.update(access='ACCESS_ERROR', error=f'{type(exc).__name__}: {exc}')
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'examples/reproduction/release/resource_links.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    if not isinstance(entries, list) or len({x['id'] for x in entries}) != len(entries):
        raise ValueError('Resource list must have unique IDs')
    if any(not x['url'].startswith('https://') for x in entries):
        raise ValueError('Resource links must use HTTPS')
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(probe, entries))
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'available': sum(x['access'] == 'AVAILABLE' for x in checks),
              'total': len(checks), 'weight_downloaded': False, 'inference_executed': False,
              'scope': 'Resource accessibility; availability alone does not establish paper-model identity.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'available': report['available'], 'total': report['total'],
                      'report': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
