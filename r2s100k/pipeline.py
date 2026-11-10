"""Prepare, run, and evaluate frozen CAT-Seg on R2S100K."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
try:
 from .taxonomy import CLASSES, BACKGROUND_RGB, NUM_CLASSES, IMAGE_SUFFIXES, DESCRIPTION_PROMPTS
except ImportError:
 from taxonomy import CLASSES, BACKGROUND_RGB, NUM_CLASSES, IMAGE_SUFFIXES, DESCRIPTION_PROMPTS

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
DEFAULT_ROOT=Path('/run/media/minhduc/D/Study/r2s100k')
VOCAB=["background outside the labeled road regions"]+[DESCRIPTION_PROMPTS[i] for i in range(1,15)]
PALETTE=np.array([BACKGROUND_RGB]+[CLASSES[i][2] for i in range(1,15)],dtype=np.uint8)
NAMES=['Background']+[CLASSES[i][0] for i in range(1,15)]

class Progress:
 def __init__(self, label, total, every=None):
  self.label=label; self.total=total; self.done=0; self.start=time.perf_counter()
  self.every=every or max(1,total//50); self.last=0
  print(f'[{label}] start ({total} items)',flush=True)
 def update(self, item=None, force=False):
  self.done+=1
  if force or self.done==self.total or self.done-self.last>=self.every:
   elapsed=time.perf_counter()-self.start; rate=self.done/elapsed if elapsed else 0
   remaining=(self.total-self.done)/rate if rate else 0
   pct=100*self.done/self.total if self.total else 100
   suffix=f' | {item}' if item else ''
   print(f'[{self.label}] {self.done}/{self.total} ({pct:.1f}%) | {elapsed:.1f}s elapsed | {rate:.2f}/s | ETA {remaining:.1f}s{suffix}',flush=True)
   self.last=self.done
 def finish(self):
  if self.total==0: print(f'[{self.label}] complete (0 items)',flush=True)

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def list_images(d): return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
def find_gt(root,split,image):
 candidates=[root/f'{split}_labels'/f'{image.name}___fuse.png',root/f'{split}_labels'/'Train-Labels'/f'{image.name}___fuse.png']
 return next((p for p in candidates if p.is_file()),None)
_RGB_LUT=np.full(1<<24,255,dtype=np.uint8)
_RGB_KEYS=(PALETTE[:,0].astype(np.uint32)<<16)|(PALETTE[:,1].astype(np.uint32)<<8)|PALETTE[:,2].astype(np.uint32)
_RGB_LUT[_RGB_KEYS]=np.arange(NUM_CLASSES,dtype=np.uint8)
def decode(rgb):
 packed=(rgb[...,0].astype(np.uint32)<<16)|(rgb[...,1].astype(np.uint32)<<8)|rgb[...,2].astype(np.uint32)
 ids=_RGB_LUT[packed]
 if np.any(ids==255):
  bad=np.unique(packed[ids==255])[:8]
  colors=[(int(x>>16),int((x>>8)&255),int(x&255)) for x in bad]
  raise ValueError(f'Unknown GT colors: {colors}')
 return ids
def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def prepare(a):
 root=a.dataset_root.resolve(); out=a.output.resolve(); report={'dataset_root':str(root),'splits':{},'cross_split_duplicate_names':{},'warnings':[]}
 names_by_split={}
 for split in ('train','val','test'):
  imgs=list_images(root/split) if (root/split).is_dir() else []
  print(f'[prepare] auditing {split}: {len(imgs)} images',flush=True)
  rec=[]; missing=[]; bad=[]
  progress=Progress(f'prepare:{split}',len(imgs))
  for im in imgs:
   gt=find_gt(root,split,im)
   rec.append({'image_id':im.name,'image_path':str(im.resolve())})
   if gt is None:
    missing.append(im.name); progress.update(im.name); continue
   with Image.open(im) as x, Image.open(gt) as y:
    if x.size!=y.size:
     bad.append({'image':im.name,'image_size':x.size,'gt_size':y.size}); progress.update(im.name); continue
    colors=y.convert('RGB').getcolors(maxcolors=256)
    if colors is None: raise ValueError(f'Mask has more than 256 distinct colors: {gt}')
    allowed={tuple(map(int,c)) for c in PALETTE}
    unknown=[tuple(map(int,color)) for _,color in colors if tuple(map(int,color)) not in allowed]
    if unknown: raise ValueError(f'Unknown GT colors in {gt}: {unknown[:8]}')
   progress.update(im.name)
  progress.finish()
  names_by_split[split]={p.name for p in imgs}
  atomic_json(out/'manifests'/f'{split}_images.json',rec)
  gtlist=[{'image_id':r['image_id'],'gt_path':str(find_gt(root,split,Path(r['image_path'])).resolve())} for r in rec if find_gt(root,split,Path(r['image_path']))]
  atomic_json(out/'manifests'/f'{split}_labels.json',gtlist)
  report['splits'][split]={'images':len(imgs),'matched_masks':len(gtlist),'missing_masks':missing,'dimension_mismatch':bad}
 for x,y in [('train','val'),('train','test'),('val','test')]:
  dup=sorted(names_by_split[x]&names_by_split[y]); report['cross_split_duplicate_names'][f'{x}_{y}']=len(dup)
 if report['cross_split_duplicate_names'].get('train_val',0): report['warnings'].append('Train and val share image basenames; inspect hashes before interpreting validation results.')
 pilot=list_images(root/'val')
 random.Random(42).shuffle(pilot); pilot=sorted(pilot[:min(20,len(pilot))])
 atomic_json(out/'manifests'/'val_pilot_images.json',[{'image_id':p.name,'image_path':str(p.resolve())} for p in pilot])
 atomic_json(out/'audit_report.json',report); atomic_json(HERE/'vocabulary.json',VOCAB)
 print('[prepare] audit complete; summary:',flush=True); print(json.dumps(report,indent=2),flush=True)
def overlay(image,ids,path):
 pal=PALETTE.copy(); pal[0]=[0,0,0]
 color=pal[ids]; blend=(np.asarray(image.convert('RGB'),dtype=np.float32)*.55+color*.45).clip(0,255).astype(np.uint8)
 Image.fromarray(blend).save(path)
def load_manifest(p):
 d=json.loads(Path(p).read_text());return d if isinstance(d,list) else d['images']
def infer(a):
 print('[infer] loading PyTorch, Detectron2 and CAT-Seg config...',flush=True)
 import torch, cv2
 from detectron2.config import get_cfg
 from detectron2.checkpoint import DetectionCheckpointer
 from detectron2.modeling import build_model
 from detectron2.projects.deeplab import add_deeplab_config
 sys.path.insert(0,str(REPO)); import cat_seg
 cfg=get_cfg();add_deeplab_config(cfg);cat_seg.add_cat_seg_config(cfg);cfg.merge_from_file(str(REPO/'configs/vitb_384.yaml'))
 vocabfile=Path(a.vocabulary).resolve();vocab=json.loads(vocabfile.read_text())
 if len(vocab)!=15 or not all(isinstance(x,str) for x in vocab): raise ValueError('Vocabulary must contain 15 ordered strings (BG then IDs 1..14).')
 cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=15;cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON=str(vocabfile);cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON=str(vocabfile);cfg.MODEL.WEIGHTS=str(a.checkpoint.resolve());cfg.MODEL.DEVICE=a.device;cfg.TEST.SLIDING_WINDOW=False;cfg.freeze()
 print(f'[infer] building model on {a.device} and loading checkpoint {a.checkpoint}',flush=True)
 model=build_model(cfg); check=DetectionCheckpointer(model).load(str(a.checkpoint.resolve()))
 # Fail closed if a checkpoint misses trained CAT-Seg predictor/aggregation weights.
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state=state.get('model',state)
 own=model.state_dict(); missing=[k for k in own if k.startswith('sem_seg_head.predictor.transformer') and k not in state]
 if missing: raise RuntimeError(f'Checkpoint lacks CAT-Seg aggregation weights, e.g. {missing[:5]}')
 model.eval();device=torch.device(a.device); model.to(device)
 print('[infer] model ready; reading image manifest...',flush=True)
 entries=load_manifest(a.manifest); entries=entries[:a.max_images] if a.max_images else entries
 print(f'[infer] processing {len(entries)} images; output: {a.output}',flush=True)
 out=Path(a.output); pred=out/'predictions';ov=out/'overlays'; pred.mkdir(parents=True,exist_ok=True);ov.mkdir(parents=True,exist_ok=True)
 source_paths={'pipeline':Path(__file__).resolve(),'taxonomy':HERE/'taxonomy.py','predictor':REPO/'cat_seg/modeling/transformer/cat_seg_predictor.py','config':REPO/'configs/vitb_384.yaml'}
 source_hashes={name:digest(path) for name,path in source_paths.items()}
 fingerprint=hashlib.sha256((digest(a.checkpoint)+digest(vocabfile)+digest(Path(a.manifest))+json.dumps(source_hashes,sort_keys=True)).encode()).hexdigest()
 try: revision=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 except Exception: revision='unknown'
 atomic_json(out/'run_metadata.json',{'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':digest(a.checkpoint),'vocabulary_sha256':digest(vocabfile),'manifest_sha256':digest(Path(a.manifest)),'fingerprint':fingerprint,'device':a.device,'config':str(REPO/'configs/vitb_384.yaml'),'source_sha256':source_hashes,'num_classes':15,'amp':False,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cat_seg_git_revision':revision})
 progress=Progress('infer',len(entries))
 for ix,r in enumerate(entries):
  im=Path(r['image_path']); dst=pred/f'{r["image_id"]}.png'; meta=pred/f'{r["image_id"]}.json'
  if dst.is_file() and meta.is_file():
   try:
    old=json.loads(meta.read_text())
    if old.get('fingerprint')==fingerprint and np.asarray(Image.open(dst)).shape==np.asarray(Image.open(im)).shape[::-1]:
     progress.update(r['image_id']); continue
   except Exception: pass
   dst.unlink(missing_ok=True);meta.unlink(missing_ok=True)
  with Image.open(im) as pil: rgb=np.asarray(pil.convert('RGB')); h,w=rgb.shape[:2]
  tensor=torch.as_tensor(rgb.astype('float32').transpose(2,0,1),device=device)
  t=time.perf_counter()
  with torch.inference_mode(): result=model([{'image':tensor,'height':h,'width':w}])[0]['sem_seg']
  scores=result.detach().float().cpu().numpy()
  if scores.shape[0]!=15 or not np.isfinite(scores).all(): raise RuntimeError(f'Invalid score map for {im}: {scores.shape}')
  scores=cv2.resize(scores.transpose(1,2,0),(w,h),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
  ids=scores.argmax(0).astype(np.uint8)
  Image.fromarray(ids,mode='L').save(dst)
  with Image.open(im) as pil: overlay(pil,ids,ov/f'{im.name}.jpg')
  if a.save_max_score: np.save(pred/f'{im.name}.max_score.npy',scores.max(0).astype(np.float16))
  atomic_json(meta,{'fingerprint':fingerprint,'image_id':im.name,'height':h,'width':w,'elapsed_seconds':time.perf_counter()-t,'status':'ok'})
  progress.update(im.name)
 progress.finish()
 print(f'[infer] complete; predictions: {pred}; overlays: {ov}',flush=True)
def erode_inner(mask):
 # 3x3 erosion; exterior image border is replicated to avoid artificial frame edges.
 import cv2
 return cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE).astype(bool)
def edge(mask): return mask & ~erode_inner(mask)
def evaluate(a):
 import cv2
 entries=load_manifest(a.image_manifest); labels={r['image_id']:Path(r['gt_path']) for r in load_manifest(a.label_manifest)}
 print(f'[evaluate] evaluating {len(entries)} images against {len(labels)} labels...',flush=True)
 predroot=Path(a.predictions); cm=np.zeros((15,15),np.int64); btp=np.zeros(15,np.int64); bp=np.zeros(15,np.int64); bg=np.zeros(15,np.int64)
 binary=np.zeros((2,2),np.int64); evaluated=0; examples=[]
 kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
 progress=Progress('evaluate',len(entries))
 for r in entries:
  name=r['image_id']; pp=predroot/f'{name}.png'
  if name not in labels or not pp.is_file(): raise FileNotFoundError(f'Missing prediction or GT for {name}')
  pr=np.asarray(Image.open(pp).convert('L')); gt=decode(np.asarray(Image.open(labels[name]).convert('RGB')))
  if pr.shape!=gt.shape: raise ValueError(f'Shape mismatch for {name}')
  if pr.max()>=15: raise ValueError(f'Invalid predicted ID in {pp}')
  cm+=np.bincount((15*gt.astype(np.int64)+pr.astype(np.int64)).ravel(),minlength=225).reshape(15,15)
  g=gt>0;p=pr>0; binary+=np.bincount((2*g.astype(np.int8)+p.astype(np.int8)).ravel(),minlength=4).reshape(2,2)
  for c in range(15):
   ge=edge(gt==c);pe=edge(pr==c);bp[c]+=pe.sum();bg[c]+=ge.sum()
   # Count matched boundary pixels within 3px; both sides independently, symmetric precision/recall.
   if pe.any() and ge.any():
    dgt=cv2.dilate(ge.astype(np.uint8),kernel)>0; dpr=cv2.dilate(pe.astype(np.uint8),kernel)>0
    btp[c]+=np.logical_and(pe,dgt).sum()+np.logical_and(ge,dpr).sum()
   if len(examples)<12 and np.any(gt!=pr): examples.append((name,gt,pr))
  evaluated+=1
  progress.update(name)
 progress.finish()
 tp=cm.diagonal(); union=cm.sum(0)+cm.sum(1)-tp
 iou=np.divide(tp,union,out=np.full(15,np.nan),where=union>0); precision=np.divide(tp,cm.sum(0),out=np.full(15,np.nan),where=cm.sum(0)>0);recall=np.divide(tp,cm.sum(1),out=np.full(15,np.nan),where=cm.sum(1)>0);dice=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.full(15,np.nan),where=(cm.sum(0)+cm.sum(1))>0)
 bf1=np.divide(btp,bp+bg,out=np.full(15,np.nan),where=(bp+bg)>0)
 def mean(x):
  v=x[np.isfinite(x)];return float(v.mean()) if v.size else None
 btpix=int(binary[1,1]); bu=int(binary[:,1].sum()+binary[1,:].sum()-btpix)
 report={'evaluated_images':evaluated,'miou_15':mean(iou),'miou_foreground14':mean(iou[1:]),'foreground_binary_iou':btpix/bu if bu else None,'foreground_binary_dice':2*btpix/(binary[:,1].sum()+binary[1,:].sum()) if binary[:,1].sum()+binary[1,:].sum() else None,'boundary_f1_foreground14':mean(bf1[1:]),'boundary_f1_tolerance_pixels':3,'boundary_metric_note':'Per-class semantic boundaries, 3x3 erosion, symmetric matching within 3px using 7x7 elliptical kernel; project supplementary metric.','class_metrics':{str(i):{'name':NAMES[i],'iou':None if not np.isfinite(iou[i]) else float(iou[i]),'dice':None if not np.isfinite(dice[i]) else float(dice[i]),'precision':None if not np.isfinite(precision[i]) else float(precision[i]),'recall':None if not np.isfinite(recall[i]) else float(recall[i]),'gt_pixels':int(cm[i].sum()),'pred_pixels':int(cm[:,i].sum()),'boundary_f1':None if not np.isfinite(bf1[i]) else float(bf1[i])} for i in range(15)}}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'metrics.json',report)
 with open(out/'class_metrics.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','name','iou','dice','precision','recall','gt_pixels','pred_pixels','boundary_f1']);w.writeheader()
  for i in range(15):w.writerow({'id':i,**report['class_metrics'][str(i)]})
 np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=','); norm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0);np.savetxt(out/'confusion_matrix_gt_normalized.csv',norm,fmt='%.6f',delimiter=',')
 # Save misclassification visual examples and GT overlays.
 exdir=out/'qualitative';exdir.mkdir(exist_ok=True)
 for name,gt,pr in examples:
  image=Image.open(next(Path(x['image_path']) for x in entries if x['image_id']==name)).convert('RGB'); overlay(image,gt,exdir/f'{name}.gt.jpg');overlay(image,pr,exdir/f'{name}.pred.jpg')
  err=np.zeros((*gt.shape,3),np.uint8);err[gt==pr]=[30,150,50];err[(gt>0)&(pr==0)]=[240,50,50];err[(gt==0)&(pr>0)]=[50,100,250];err[(gt>0)&(pr>0)&(gt!=pr)]=[255,190,0];Image.fromarray(err).save(exdir/f'{name}.errors.png')
 print('[evaluate] complete; metrics:',flush=True); print(json.dumps(report,indent=2),flush=True)
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
 q=sp.add_parser('prepare');q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--output',type=Path,default=HERE/'runs/r2s100k');q.set_defaults(fn=prepare)
 q=sp.add_parser('infer');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--vocabulary',type=Path,default=HERE/'vocabulary.json');q.add_argument('--output',type=Path,required=True);q.add_argument('--device',default='cuda');q.add_argument('--max-images',type=int,default=0);q.add_argument('--save-max-score',action='store_true');q.set_defaults(fn=infer)
 q=sp.add_parser('evaluate');q.add_argument('--image-manifest',type=Path,required=True);q.add_argument('--label-manifest',type=Path,required=True);q.add_argument('--predictions',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.set_defaults(fn=evaluate)
 a=p.parse_args();a.fn(a)
if __name__=='__main__':main()
