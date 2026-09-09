"""Download a fixed public EVA checkpoint and verify its published LFS checksum."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = '514db6705637c1ec963b728768fc9b34728699ee'
CHECKSUMS = {
    'EVA_21M': '45d56a7399c4936429da149edf5003bbb5999490a64ec7d793bf5ac98116eb01',
    'EVA_1.4B_GLM': '1ee172fe1bf68c6907590dae630b6e1d8493225dd0b5a34ecc0ead0cc51d4f0c',
    'EVA_1.4B_CLM': '323c13d571d0be87e450cb7b103bf8b28396e55e877519e2d2fb3c434875d420',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=CHECKSUMS, default='EVA_21M')
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    files = {}
    for name in ['config.json', 'tokenizer.json', 'model_weights.pt']:
        path = hf_hub_download('GENTEL-Lab/EVA', f'{args.model}/{name}',
                              revision=REVISION, local_dir=args.destination)
        files[name] = sha256(path)
    if files['model_weights.pt'] != CHECKSUMS[args.model]:
        raise ValueError('Checkpoint checksum differs from pinned public LFS object')
    manifest = dict(repo='GENTEL-Lab/EVA', revision=REVISION, model=args.model, sha256=files)
    dest = args.destination / args.model / 'download_manifest.json'
    dest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
