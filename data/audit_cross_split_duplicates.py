"""Recompute the cross-split 64-bit difference-hash candidates.

Read-only: writes no files, and compares computed candidates with the separately
saved visual-review record. Each source image is decoded as luminance and
resized to 9x8; comparisons use right pixel > left pixel. This detects only
some near-duplicate photos and does not establish animal-level independence.
"""
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
splits = json.loads((root / 'splits_valid.json').read_text(encoding='utf-8'))
trainval = splits['train'] + splits['val']
test = splits['test']
ids = trainval + test
hashes = np.empty((len(ids), 8), dtype=np.uint8)
for i, stem in enumerate(ids):
    with Image.open(root / 'oxford-iiit-pet' / 'images' / f'{stem}.jpg') as image:
        pixels = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    hashes[i] = np.packbits((pixels[:, 1:] > pixels[:, :-1]).reshape(-1))
lookup = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
pairs = []
for i in range(len(trainval)):
    dist = lookup[np.bitwise_xor(hashes[i], hashes[len(trainval):])].sum(axis=1)
    for j in np.where(dist <= 5)[0]:
        pairs.append({'trainval': ids[i], 'test': test[int(j)], 'distance': int(dist[j])})
pairs.sort(key=lambda x: (x['distance'], x['trainval'], x['test']))
review = json.loads((root / 'cross_split_dhash_review.json').read_text(encoding='utf-8'))
expected = [{k: p[k] for k in ('trainval', 'test', 'distance')} for p in review['pairs']]
if pairs != expected:
    raise RuntimeError('Computed candidates differ from saved manual review; inspect changes before use.')
print(json.dumps({'trainval': len(trainval), 'test': len(test), 'candidate_pairs': len(pairs),
                  'manual_near_duplicate_pairs': sum(p['review'] == 'near_duplicate' for p in review['pairs']),
                  'manual_false_positives': sum(p['review'] == 'different_photo' for p in review['pairs'])}, indent=2))
