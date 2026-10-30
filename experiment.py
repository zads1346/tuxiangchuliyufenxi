"""Oxford-IIIT Pet paired U-Net experiment. Upstream U-Net: milesial/Pytorch-UNet, GPL-3.0."""
import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream'))
from unet import UNet

class OxfordPet(Dataset):
    def __init__(self, root, ids, size, augment=False):
        self.root, self.ids, self.size, self.augment = Path(root), ids, size, augment
    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        stem = self.ids[index]
        with Image.open(self.root / 'images' / (stem + '.jpg')) as image:
            image = np.asarray(image.convert('RGB').resize((self.size,self.size), Image.Resampling.BILINEAR),dtype=np.float32)/255.0
        with Image.open(self.root / 'annotations' / 'trimaps' / (stem + '.png')) as mask:
            mask = np.asarray(mask.resize((self.size,self.size), Image.Resampling.NEAREST))
            labels = set(np.unique(mask).tolist())
            if not labels.issubset({1, 2, 3}): raise ValueError(f'Unexpected trimap labels for {stem}: {labels}')
        if self.augment:
            # Flip image and trimap together; brightness applies only to the image.
            if random.random() < .5: image, mask = np.flip(image, 1), np.flip(mask, 1)
            image = np.clip(image * random.uniform(0.85, 1.15), 0.0, 1.0)
        labels = torch.from_numpy(mask.copy())
        target = (labels == 1).float()
        valid = (labels != 3).float()
        return torch.from_numpy(image.copy()).permute(2,0,1), target, valid, stem

def scores(logits, masks, valid):
    pred=(torch.sigmoid(logits.squeeze(1)) >= .5).float()
    tp=(pred*masks*valid).sum(dim=(1,2)); p=(pred*valid).sum(dim=(1,2)); t=(masks*valid).sum(dim=(1,2))
    dice=(2*tp+1e-6)/(p+t+1e-6); iou=(tp+1e-6)/(p+t-tp+1e-6)
    return dice, iou, pred

def masked_loss(logits, targets, valid):
    # Label 3 (ambiguous boundary) has no contribution to either loss term.
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probability = torch.sigmoid(logits) * valid
    targets = targets * valid
    dice = (2 * (probability * targets).sum() + 1e-6) / (probability.sum() + targets.sum() + 1e-6)
    return bce + (1 - dice)

def evaluate(model, loader, device, save=None):
    model.eval(); rows=[]
    if save: (save/'predictions').mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for x,y,valid,ids in loader:
            x,y,valid=x.to(device),y.to(device),valid.to(device); logits=model(x); dice,iou,pred=scores(logits,y,valid)
            for j,stem in enumerate(ids):
                rows.append({'id':stem,'dice':float(dice[j]),'iou':float(iou[j])})
                if save:
                    Image.fromarray((pred[j].cpu().numpy()*255).astype(np.uint8)).save(save/'predictions'/f'{stem}.png')
    return {'dice':float(np.mean([r['dice'] for r in rows])), 'iou':float(np.mean([r['iou'] for r in rows])), 'n':len(rows)},rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',required=True,choices=['baseline','augmented']); p.add_argument('--data',type=Path,default=ROOT/'data'/'oxford-iiit-pet'); p.add_argument('--split',type=Path,default=ROOT/'data'/'splits_valid.json'); p.add_argument('--out',type=Path,default=ROOT/'runs'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=4); p.add_argument('--size',type=int,default=256); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=42); p.add_argument('--workers',type=int,default=0); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device:',device,flush=True)
    split=json.loads(args.split.read_text(encoding='utf-8')); ids={k:split[k] for k in ('train','val','test')}
    if any(not ids[k] for k in ids) or len(set(sum(ids.values(),[]))) != sum(map(len,ids.values())): raise ValueError('Empty or overlapping split')
    ds={k:OxfordPet(args.data,ids[k],args.size,args.variant=='augmented' and k=='train') for k in ids}
    train_gen=torch.Generator().manual_seed(args.seed)
    train=DataLoader(ds['train'],batch_size=args.batch,shuffle=True,generator=train_gen,num_workers=args.workers,pin_memory=device.type=='cuda')
    val=DataLoader(ds['val'],batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
    test=DataLoader(ds['test'],batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
    out=args.out/args.variant; out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy(); config.update(device=str(device),torch=torch.__version__); config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}; (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    model=UNet(n_channels=3,n_classes=1,bilinear=True).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr); best=-1; history=[]; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0; count=0
        for x,y,valid,_ in train:
            x,y,valid=x.to(device),y.to(device),valid.to(device); opt.zero_grad(set_to_none=True); logits=model(x).squeeze(1); loss=masked_loss(logits,y,valid); loss.backward(); opt.step(); total+=loss.item()*len(x); count+=len(x)
        v,_=evaluate(model,val,device); row={'epoch':epoch,'train_loss':total/count,'val_dice':v['dice'],'val_iou':v['iou']}; history.append(row)
        if v['dice']>best:
            best=v['dice']; torch.save({'model':model.state_dict(),'epoch':epoch,'val_dice':best,'config':config},out/'best.pt')
        with (out/'history.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row.keys()); writer.writeheader(); writer.writerows(history)
        print(f"{args.variant} epoch {epoch:02d}/{args.epochs} loss={row['train_loss']:.4f} val_dice={v['dice']:.4f} val_iou={v['iou']:.4f} elapsed={(time.perf_counter()-started):.0f}s",flush=True)
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(checkpoint['model'])
    result,rows=evaluate(model,test,device,save=out); result.update({'best_epoch':checkpoint['epoch'],'best_val_dice':checkpoint['val_dice'],'seconds_training_and_eval':time.perf_counter()-started,'threshold':0.5,'metric_aggregation':f'mean of per-image scores on valid pixels, {args.size}x{args.size} masks, label 3 ignored'})
    (out/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'test_per_image.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','dice','iou']); writer.writeheader(); writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]); axes[0].set(xlabel='Epoch',ylabel='BCE + Dice loss',title='Training loss'); axes[1].plot([r['epoch'] for r in history],[r['val_dice'] for r in history],label='Dice'); axes[1].plot([r['epoch'] for r in history],[r['val_iou'] for r in history],label='IoU'); axes[1].set(xlabel='Epoch',ylabel='Score',title='Validation'); axes[1].legend(); fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=150); plt.close(fig)
    print('TEST',json.dumps(result),flush=True)
if __name__=='__main__': main()
