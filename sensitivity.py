"""Evaluate pre-reviewed cross-split near-duplicate exclusions without retraining.

Uses image-only review recorded before seeing model predictions. Writes aggregate
results only after validating test membership and both per-image metric files.
"""
import csv, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read_json(path):
    with path.open(encoding='utf-8') as f: return json.load(f)

def mean(values):
    values = list(values)
    if not values: raise ValueError('Cannot average an empty set')
    return math.fsum(values)/len(values)

def main():
    splits=read_json(ROOT/'data/splits_valid.json'); original=splits['test']; original_set=set(original)
    if len(original_set)!=len(original): raise ValueError('Duplicate identifiers in test split')
    excluded=read_json(ROOT/'data/duplicate_test_ids.json'); ids=excluded['test_ids']; exclude_set=set(ids)
    if len(exclude_set)!=len(ids) or len(ids)!=int(excluded['n']): raise ValueError('Invalid excluded-ID count or duplicates')
    if not exclude_set <= original_set: raise ValueError(f'Excluded IDs not in test set: {sorted(exclude_set-original_set)}')
    review=read_json(ROOT/'data/cross_split_dhash_review.json')
    reviewed={p['test'] for p in review['pairs'] if p['review']=='near_duplicate'}
    if reviewed!=exclude_set: raise ValueError('Excluded IDs do not match image-only manual review')
    if not all(p['trainval'] in set(splits['train']+splits['val']) and p['test'] in original_set for p in review['pairs']): raise ValueError('Review contains an ID outside the audited splits')
    summary={'method':'Image-only cross-split dHash candidate review; exclude manually confirmed near-duplicate test IDs from evaluation only; no retraining.', 'exclude_source':'data/duplicate_test_ids.json', 'review_source':'data/cross_split_dhash_review.json', 'excluded_test_ids':ids, 'n_excluded':len(ids), 'limitations':review['limitations'], 'results':{}}
    rows_out=[]
    for name in ('baseline','augmented'):
        d=ROOT/'runs'/name; reported=read_json(d/'test_metrics.json')
        with (d/'test_per_image.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        scores={}
        for row in rows:
            sid=row['id']; dice=float(row['dice']); iou=float(row['iou'])
            if sid in scores: raise ValueError(f'{name}: duplicate test ID {sid}')
            if any(not math.isfinite(v) or not 0<=v<=1 for v in (dice,iou)): raise ValueError(f'{name}: invalid score {sid}')
            scores[sid]=(dice,iou)
        if set(scores)!=original_set or len(scores)!=int(reported['n']): raise ValueError(f'{name}: test metric IDs do not match fixed split')
        all_dice=mean(scores[x][0] for x in original); all_iou=mean(scores[x][1] for x in original)
        if abs(all_dice-float(reported['dice']))>1e-8 or abs(all_iou-float(reported['iou']))>1e-8: raise ValueError(f'{name}: per-image means disagree with reported test metrics')
        kept=[x for x in original if x not in exclude_set]
        clean_dice=mean(scores[x][0] for x in kept);clean_iou=mean(scores[x][1] for x in kept)
        result={'official_valid_subset':{'n':len(original),'dice':all_dice,'iou':all_iou},'after_near_duplicate_exclusion':{'n':len(kept),'dice':clean_dice,'iou':clean_iou},'delta_excluded_minus_original':{'dice':clean_dice-all_dice,'iou':clean_iou-all_iou}}
        summary['results'][name]=result
        for category in ('official_valid_subset','after_near_duplicate_exclusion'):
            rows_out.append({'variant':name,'subset':category,**result[category]})
    out=ROOT/'runs'/'sensitivity'; out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','subset','n','dice','iou']);writer.writeheader();writer.writerows(rows_out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
