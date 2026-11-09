"""Reconstruct the wing-root trial RE2 from the accepted quadratic baseline."""
import argparse
import gzip
import hashlib
from pathlib import Path
import struct
import tempfile

HEADER = 84
RECORD = 200
POINT = struct.Struct('<I3d')
FORMAT = b'NMP-RE2-PATCH-v1\0'


def apply_patch(baseline, patch, output):
    if output.resolve() == baseline.resolve():
        raise ValueError('Choose a different output path from the baseline.')
    data = baseline.read_bytes()
    with gzip.open(patch, 'rb') as f:
        if f.read(len(FORMAT)) != FORMAT:
            raise ValueError('Unknown mesh patch format.')
        old_sha, new_sha = f.read(32), f.read(32)
        if hashlib.sha256(data).digest() != old_sha:
            raise ValueError('Baseline checksum differs from the accepted quadratic mesh.')
        elements, npoints = struct.unpack('<II', f.read(8))
        entries = f.read()
    if len(entries) != npoints * POINT.size or HEADER + elements * RECORD > len(data):
        raise ValueError('Invalid mesh patch size.')

    def offsets(corner):
        e, i = divmod(corner, 8)
        if e >= elements:
            raise ValueError('Patch corner index exceeds the mesh size.')
        p = HEADER + e * RECORD + 8 + 8 * i
        return p, p + 64, p + 128

    def xyz(p):
        return tuple(data[i:i+8] for i in p)

    edits = {}
    for corner, *coords in POINT.iter_unpack(entries):
        edits[xyz(offsets(corner))] = tuple(struct.pack('<d', c) for c in coords)

    # The wing-root edits occupy the first six radial layers. Matching old
    # corner coordinates propagates each edit to every incident element.
    result = bytearray(data)
    for corner in range(min(elements, 6 * 10856) * 8):
        off = offsets(corner)
        new = edits.get(xyz(off))
        if new is not None:
            for i, value in zip(off, new):
                result[i:i+8] = value
    if hashlib.sha256(result).digest() != new_sha:
        raise ValueError('Reconstructed mesh does not match the validated trial.')
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(result)
            f.flush()
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    return new_sha.hex()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--patch', type=Path, default=Path(__file__).with_name('aircraft_wing_root.re2.patch.gz'))
    args = parser.parse_args()
    print(apply_patch(args.baseline, args.patch, args.output))
