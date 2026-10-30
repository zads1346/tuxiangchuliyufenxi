"""Make fixed-sample side-by-side comparisons once both experiment variants finish."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
split=json.loads((ROOT/'data/splits_valid.json').read_text(encoding='utf-8'))
base=ROOT/'runs/baseline'; aug=ROOT/'runs/augmented'; out=ROOT/'runs/comparison'; out.mkdir(parents=True,exist_ok=True)
metrics=[]
for name,path in [('baseline',base),('augmented',aug)]:
    m=json.loads((path/'test_metrics.json').read_text()); metrics.append({'variant':name,'dice':m['dice'],'iou':m['iou'],'n':m['n'],'best_epoch':m['best_epoch']})
with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=metrics[0].keys());w.writeheader();w.writerows(metrics)
# Fixed indices from the saved test list; not chosen by model performance.
ids=[split['test'][i*len(split['test'])//6] for i in range(6)]
fig,axes=plt.subplots(len(ids),4,figsize=(12,3*len(ids)))
for i,stem in enumerate(ids):
    paths=[ROOT/'data/oxford-iiit-pet/images'/f'{stem}.jpg',ROOT/'data/oxford-iiit-pet/annotations/trimaps'/f'{stem}.png',base/'predictions'/f'{stem}.png',aug/'predictions'/f'{stem}.png']
    for j,path in enumerate(paths):
        with Image.open(path) as image:
            if j==0: arr=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)); axes[i,j].imshow(arr)
            elif j==1: arr=np.asarray(image.resize((256,256),Image.Resampling.NEAREST)); arr=np.where(arr==1,1.0,np.where(arr==3,0.5,0.0)); axes[i,j].imshow(arr,cmap='gray',vmin=0,vmax=1)
            else: arr=np.asarray(image.convert('L').resize((256,256),Image.Resampling.NEAREST))>127; axes[i,j].imshow(arr,cmap='gray',vmin=0,vmax=1)
        axes[i,j].axis('off')
        if i==0: axes[i,j].set_title(['Input','GT (gray = ignore)','Baseline','Augmentation'][j])
    axes[i,0].text(0,-.05,stem,transform=axes[i,0].transAxes,fontsize=7)
fig.tight_layout();fig.savefig(out/'fixed_examples.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for name,path in [('Baseline',base),('Augmented',aug)]:
    with (path/'history.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    axes[0].plot([int(x['epoch']) for x in rows],[float(x['train_loss']) for x in rows],label=name)
    axes[1].plot([int(x['epoch']) for x in rows],[float(x['val_dice']) for x in rows],label=name)
for ax,y,title in [(axes[0],'Loss','Training loss'),(axes[1],'Dice','Validation Dice')]:ax.set(xlabel='Epoch',ylabel=y,title=title);ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'training_comparison.png',dpi=150);plt.close(fig)
print(json.dumps(metrics,indent=2)); print('Fixed test examples:',', '.join(ids))
