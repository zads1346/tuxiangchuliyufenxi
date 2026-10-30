"""Ablate flip TTA and small-component filtering with validation-only tuning.

All variants use the already trained baseline checkpoint. Test set is read only
after validation has fixed the component threshold. No retraining is performed.
"""
import csv
import hashlib
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OxfordPet, UNet

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'oxford-iiit-pet'
OUT = ROOT / 'runs' / 'inference_improvements'
CANDIDATES = [0, 16, 64, 256, 1024]

def metrics(pred, target, valid):
    p, t, v = pred.astype(bool), target.astype(bool), valid.astype(bool)
    n_p = int(np.count_nonzero(p & v))
    n_t = int(np.count_nonzero(t & v))
    inter = int(np.count_nonzero(p & t & v))
    return (2*inter + 1e-6)/(n_p+n_t+1e-6), (inter + 1e-6)/(n_p+n_t-inter+1e-6)

def remove_small(mask, minimum):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return keep[labels].astype(bool)

def avg(rows):
    return {'n': len(rows), 'dice': float(np.mean([r['dice'] for r in rows])),
            'iou': float(np.mean([r['iou'] for r in rows]))}

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'dice', 'iou'])
        w.writeheader(); w.writerows(rows)

def infer(loader, model, device):
    with torch.no_grad():
        for images, targets, valid, ids in loader:
            images = images.to(device)
            original = torch.sigmoid(model(images)).squeeze(1)
            flipped = torch.sigmoid(model(torch.flip(images, dims=[3]))).squeeze(1)
            flipped = torch.flip(flipped, dims=[2])
            tta = (original + flipped) * .5
            for i, stem in enumerate(ids):
                yield stem, (original[i] >= .5).cpu().numpy(), (tta[i] >= .5).cpu().numpy(), targets[i].numpy(), valid[i].numpy()

def main():
    started = time.perf_counter()
    cv2.setNumThreads(1); torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = json.loads((ROOT/'data/splits_valid.json').read_text(encoding='utf-8'))
    assert not (set(splits['val']) & set(splits['test']))
    checkpoint_path = ROOT/'runs/baseline/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    model.load_state_dict(checkpoint['model']); model.eval()
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    val_loader = DataLoader(OxfordPet(DATA, splits['val'], 256), batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(OxfordPet(DATA, splits['test'], 256), batch_size=4, shuffle=False, num_workers=0)
    val_rows = {n: [] for n in CANDIDATES}
    val_tta = []
    print('Validation: select minimum connected-component area using mean per-image Dice', flush=True)
    for stem, base, tta, y, v in infer(val_loader, model, device):
        d, i = metrics(tta, y, v); val_tta.append({'id':stem,'dice':d,'iou':i})
        for n in CANDIDATES:
            d, i = metrics(remove_small(tta,n), y, v); val_rows[n].append({'id':stem,'dice':d,'iou':i})
    # Choose by validation Dice, with the smaller threshold winning exact ties.
    selected = max(CANDIDATES, key=lambda n:(avg(val_rows[n])['dice'], -n))
    validation = {'tta_without_filter':avg(val_tta), 'thresholds':{str(n):avg(val_rows[n]) for n in CANDIDATES},
                  'chosen_minimum_area_pixels':selected, 'rule':'maximum validation mean Dice; tie -> smaller threshold'}
    (OUT/'validation_selection.json').write_text(json.dumps(validation, indent=2),encoding='utf-8')
    for n in CANDIDATES: write_csv(OUT/f'validation_min_area_{n}.csv',val_rows[n])
    print('Validation selected', selected, json.dumps(validation), flush=True)
    # Threshold frozen before any test inference.
    test_rows = {k: [] for k in ['baseline', 'tta', 'tta_components']}
    fixed = [splits['test'][i*len(splits['test'])//6] for i in range(6)]
    fixed_images = {}
    print('Testing frozen rule', flush=True)
    for j,(stem,base,tta,y,v) in enumerate(infer(test_loader, model, device)):
        filtered = remove_small(tta,selected)
        for key,pred in [('baseline',base), ('tta',tta), ('tta_components',filtered)]:
            d,i=metrics(pred,y,v);test_rows[key].append({'id':stem,'dice':d,'iou':i})
        if stem in fixed: fixed_images[stem] = (base, tta, filtered)
        if (j+1)%500==0: print('Test',j+1,'/',len(splits['test']),flush=True)
    # Cross-check the metrics from the current local training run, not the
    # archived results from another machine. Numerical kernels can differ.
    baseline_path = ROOT/'runs/baseline/test_metrics.json'
    if baseline_path.exists():
        baseline_current = json.loads(baseline_path.read_text(encoding='utf-8'))
        for key in ['dice', 'iou']:
            if abs(avg(test_rows['baseline'])[key] - baseline_current[key]) > 1e-6:
                warnings.warn(f'Baseline {key} differs from the current runs/baseline/test_metrics.json; verify the checkpoint, data split and execution environment.')
    else:
        warnings.warn('Current baseline test_metrics.json is missing; no baseline consistency check was performed.')
    excluded = set(json.loads((ROOT/'data/duplicate_test_ids.json').read_text(encoding='utf-8'))['test_ids'])
    assert excluded <= set(splits['test'])
    summary={'checkpoint':str(checkpoint_path.relative_to(ROOT)),'checkpoint_sha256':checkpoint_sha256,
             'device':str(device),'image_size':256,'prediction_threshold':0.5,'candidate_minimum_areas':CANDIDATES,
             'selection':validation,'method':'TTA = average of original probability and horizontally flipped-input prediction flipped back; component filter removes 8-connected foreground components smaller than selected size.',
             'metrics':'per-image mean on valid pixels; original trimap label 3 ignored; filter operates on full prediction before metric masking',
             'test':{k:avg(v) for k,v in test_rows.items()},
             'test_without_reviewed_duplicates':{k:avg([r for r in v if r['id'] not in excluded]) for k,v in test_rows.items()},
             'excluded_reviewed_duplicate_test_images':len(excluded), 'elapsed_seconds':time.perf_counter()-started,
             'fixed_examples':fixed}
    for key, rows in test_rows.items(): write_csv(OUT/f'test_{key}_per_image.csv',rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(6,5,figsize=(15,18))
    for r, stem in enumerate(fixed):
        with Image.open(DATA/'images'/f'{stem}.jpg') as im: image=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        with Image.open(DATA/'annotations'/'trimaps'/f'{stem}.png') as im: mask=np.asarray(im.resize((256,256),Image.Resampling.NEAREST))
        gt=np.where(mask==1,1.0,np.where(mask==3,0.5,0.0))
        for c, arr in enumerate([image,gt,*fixed_images[stem]]):
            axes[r,c].imshow(arr) if c==0 else axes[r,c].imshow(arr,cmap='gray',vmin=0,vmax=1)
            axes[r,c].axis('off')
            if r==0: axes[r,c].set_title(['Image','GT (gray=ignore)','Baseline','Flip TTA','TTA + components'][c])
        axes[r,0].text(0,-.05,stem,transform=axes[r,0].transAxes,fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'fixed_examples.png',dpi=140);plt.close(fig)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
