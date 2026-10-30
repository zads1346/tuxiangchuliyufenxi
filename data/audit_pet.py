"""Audit Oxford-IIIT Pet and create breed-stratified splits from official lists."""
import hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'oxford-iiit-pet'
ANNO = DATA / 'annotations'

def load_list(name):
    rows = []
    for line in (ANNO / name).read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith('#'):
            rows.append((fields[0], int(fields[1])))
    return rows

official_trainval = load_list('trainval.txt')
official_test = load_list('test.txt')
groups = defaultdict(list)
for stem, breed in official_trainval:
    groups[breed].append(stem)
randomizer = random.Random(42)
train, val = [], []
for breed, stems in sorted(groups.items()):
    stems.sort()
    randomizer.shuffle(stems)
    n = max(1, round(len(stems) * .1))
    val += stems[:n]
    train += stems[n:]
test = [stem for stem, breed in official_test]
splits = {'seed': 42, 'method': '10 percent per breed from official trainval; official test retained', 'train': train, 'val': val, 'test': test}
assert len(set(train + val + test)) == len(train) + len(val) + len(test)
issues = []
labels = Counter()
sha_groups = defaultdict(list)
empty_masks = []
for subset in ('train', 'val', 'test'):
    for stem in splits[subset]:
        ip = DATA/'images'/f'{stem}.jpg'
        mp = ANNO/'trimaps'/f'{stem}.png'
        try:
            data = ip.read_bytes()
            sha_groups[hashlib.sha256(data).hexdigest()].append((subset,stem))
            with Image.open(ip) as im:
                im.load()
                size = im.size
            with Image.open(mp) as m:
                m.load()
                if m.size != size: issues.append([stem, 'size mismatch', size, m.size])
                colors = m.getcolors(maxcolors=256)
                if colors is None: issues.append([stem, 'mask has more than 256 colors'])
                else:
                    values = {v for _, v in colors}
                    labels.update(values)
                    if values == {2}: empty_masks.append({'split': subset, 'id': stem})
        except Exception as exc:
            issues.append([stem, type(exc).__name__, str(exc)])
def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

dups = [v for v in sha_groups.values() if len(v)>1]
cross = [v for v in dups if len(set(k for k,_ in v))>1]
audit = {'dataset': 'Oxford-IIIT Pet', 'official_source': 'https://www.robots.ox.ac.uk/~vgg/data/pets/', 'source_archives': ['https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz','https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'], 'archive_sha256': {p.name:file_hash(p) for p in (ROOT/'images.tar.gz', ROOT/'annotations.tar.gz')}, 'usage': 'Research purposes only; original websites terms apply, per dataset README.', 'counts': {k:len(splits[k]) for k in ('train','val','test')}, 'breed_count':len(groups), 'mask_label_image_counts':dict(labels), 'mask_legend':'1 foreground, 2 background, 3 not classified', 'invalid_or_mismatch': issues, 'exact_duplicates':dups, 'cross_split_exact_duplicates':cross, 'note':'Exact duplicate check cannot exclude near-duplicates or same animal across split.'}
empty_ids = {entry['id'] for entry in empty_masks}
valid = {**splits, 'method': splits['method'] + '; excluded all-background trimaps', 'excluded_ids': empty_masks}
for key in ('train', 'val', 'test'):
    valid[key] = [stem for stem in splits[key] if stem not in empty_ids]
audit['empty_masks'] = empty_masks
audit['empty_mask_note'] = 'All-background annotations; corresponding images visually reviewed and contain visible pets. Excluded from segmentation experiments.'
audit['valid_counts'] = {k: len(valid[k]) for k in ('train','val','test')}
(ROOT/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
(ROOT/'splits_valid.json').write_text(json.dumps(valid,indent=2),encoding='utf-8')
(ROOT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
