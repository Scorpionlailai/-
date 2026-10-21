"""Read-only source binding check; unresolved expected hashes never pass."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-map', type=Path, help='JSON original absolute path -> relocated absolute path')
    parser.add_argument('--dependency-lock', type=Path, help='JSON: {provenance: ..., hashes: {original_path: expected_sha256}}')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    mapping = json.loads(args.path_map.read_text()) if args.path_map else {}
    lock = json.loads(args.dependency_lock.read_text()) if args.dependency_lock else {}
    if lock and not lock.get('provenance'):
        raise ValueError('dependency lock must explain original source-lock provenance or explicitly newly locked revision')
    rows = []
    for asset in json.loads((root / 'assets.json').read_text()):
        original = asset['original_path']
        path = Path(mapping.get(original, original))
        expected = asset['expected_sha256'] or lock.get('hashes', {}).get(original)
        actual = None
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(2**20), b''):
                    h.update(block)
            actual = h.hexdigest()
        status = 'missing' if actual is None else 'unknown_expected_hash' if expected is None else 'pass' if actual == expected else 'hash_mismatch'
        rows.append(dict(**asset, resolved_path=str(path), resolved_expected_sha256=expected, actual_sha256=actual, status=status))
    result = dict(passed=all(r['status'] == 'pass' for r in rows), dependency_lock_provenance=lock.get('provenance'), checks=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'passed': result['passed'], 'counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
