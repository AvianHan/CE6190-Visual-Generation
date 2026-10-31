"""Fresh common-configuration calibration and conversion of official Sen raw data.

MagCache: mean of official token-wise residual magnitude ratios on independent
full-compute generated prompts. SenCache: official finite-difference functions on
independent real videos, then explicit arithmetic mean over videos. This supplies
the converter missing from the upstream script; it is an experimental convention,
not a claim that the authors' released profile used this exact aggregation.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, time
from pathlib import Path
from .policies import config_signature,validate_row,canonical_prompt

def read_samples(path,method):
 samples=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
 if not samples:raise ValueError('Empty calibration set')
 ids=set();prompts=set()
 for s in samples:
  if s.get('split')!='calibration':raise ValueError('Every row must explicitly have split=calibration')
  if not s.get('prompt_id') or not s.get('prompt'):raise ValueError('Every calibration sample needs prompt_id and prompt')
  if s['prompt_id'] in ids or canonical_prompt(s['prompt']) in prompts:raise ValueError('Duplicate calibration sample')
  ids.add(s['prompt_id']);prompts.add(canonical_prompt(s['prompt']))
  if method=='sencache' and not s.get('video_path'):raise ValueError('SenCache needs actual independent calibration videos')
 return samples

class MagnitudeCollector:
 def __init__(self,model):
  self.before=None;self.previous=[None,None];self.ratios=[]
  self.a=model.blocks[0].register_forward_pre_hook(self.start)
  self.b=model.blocks[-1].register_forward_hook(self.end)
 def start(self,module,inputs):self.before=inputs[0].detach().clone()
 def end(self,module,inputs,output):
  residual=(output-self.before).detach();branch=len(self.ratios)%2
  prev=self.previous[branch]
  if prev is None:ratio=1.0
  else:
   denom=prev.norm(dim=-1)
   if (denom==0).any():raise ValueError('Zero residual norm encountered; cannot fit official MagCache ratio')
   ratio=(residual.norm(dim=-1)/denom).mean().item()
   if not math.isfinite(ratio) or ratio<=0:raise ValueError('Nonfinite calibration magnitude ratio')
  self.ratios.append(ratio);self.previous[branch]=residual.clone()
 def close(self):self.a.remove();self.b.remove()

def load_video(path,row,device):
 import numpy as np
 import torch
 import torch.nn.functional as F
 path=Path(path)
 if path.suffix=='.npy':frames=np.load(path,allow_pickle=False)
 else:
  import imageio.v3 as iio
  frames=iio.imread(path)
 if frames.ndim!=4 or frames.shape[-1]!=3 or frames.dtype!=np.uint8:raise ValueError(f'{path}: expected uint8 T,H,W,3 RGB')
 indices=np.linspace(0,len(frames)-1,row['num_frames'],dtype=int)
 x=torch.from_numpy(frames[indices].copy()).to(device=device,dtype=torch.float32).permute(0,3,1,2)
 x=F.interpolate(x,size=(row['height'],row['width']),mode='bilinear',align_corners=False,antialias=True)
 return (x/127.5-1).permute(1,0,2,3)

def run_mag(backend,row,samples):
 import numpy as np
 profiles=[]
 for sample in samples:
  r=dict(row,method='baseline',prompt=sample['prompt'],prompt_id=sample['prompt_id'],seed=int(sample.get('seed',0)))
  collector=MagnitudeCollector(backend.pipe.model)
  try:backend.generate(r,decode=False);profiles.append(collector.ratios)
  finally:collector.close()
 profile=np.asarray(profiles,dtype=np.float64)
 if profile.shape!=(len(samples),2*row['num_inference_steps']):raise RuntimeError('Calibration shape mismatch')
 return {'mag_ratios':profile.mean(axis=0).tolist(),'raw_ratios':profile.tolist(),
   'aggregation':'arithmetic mean of per-sample token-mean ratios; not interpolated across schedules'}

def run_sen(backend,row,samples):
 import numpy as np
 import torch
 from .wan_adapter import extract_function
 source=backend.vendor/'sencache'/'Wan2.1'/'sensitivity_calculation.py'
 namespace={'torch':torch}
 f_x,digest=extract_function(source,'calculate_solver_step_sensitivity',namespace)
 f_t,_=extract_function(source,'calculate_jacobian_norm_T',namespace)
 backend.pipe.model.forward=backend.original_forward
 if backend.trace is not None:backend.trace.close();backend.trace=None
 raw=[];times=None
 for si,sample in enumerate(samples):
  scheduler=backend.make_scheduler(row)
  times=scheduler.timesteps.detach().float().cpu().numpy()
  sigmas=scheduler.sigmas.detach().float().cpu().numpy()
  if len(sigmas)<len(times):raise ValueError('Scheduler missing sigmas')
  video=load_video(sample['video_path'],row,backend.pipe.device)
  with torch.no_grad():
   clean=backend.pipe.vae.encode([video])[0]
   context,_=backend.encode_text(sample['prompt'],row.get('negative_prompt'))
  shape,seq_len=backend.shape(row)
  if tuple(clean.shape)!=shape:raise ValueError(f'Encoded latent geometry {tuple(clean.shape)} != {shape}')
  y={'context':context,'seq_len':seq_len};sample_raw=[]
  g=torch.Generator(device=backend.pipe.device).manual_seed(int(sample.get('seed',0)))
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for i,(tv,sigma) in enumerate(zip(times,sigmas)):
    t=torch.tensor([float(tv)],device=backend.pipe.device,dtype=torch.float32)
    tn=torch.tensor([float(times[i+1]) if i+1<len(times) else 0.0],device=backend.pipe.device,dtype=torch.float32)
    noise=torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=g)
    xt=(1-float(sigma))*clean+float(sigma)*noise
    jx=f_x(backend.pipe.model,scheduler,xt,t,tn,y,backend.pipe.device,int(sample.get('seed',0))+i)
    jt=f_t(backend.pipe.model,xt,t,y,tn,backend.pipe.device)
    if not math.isfinite(jx) or not math.isfinite(jt) or jx<0 or jt<0:raise ValueError('Invalid finite-difference sensitivity; abort rather than writing zeros')
    sample_raw.append([jx,jt])
  raw.append(sample_raw)
  del video,clean,context
 return aggregate_sen(np.asarray(raw),times)|{'calibration_code_sha256':digest,
   'geometry_note':'Frames sampled uniformly; resize explicitly uses (height,width), correcting upstream ambiguous W/H naming',
   'estimator_note':'Unmodified official finite-difference functions; noisy calibration states independently drawn at each time, solver history carried across times as upstream'}

def aggregate_sen(raw,timesteps):
 import numpy as np
 raw=np.asarray(raw,dtype=np.float64);times=np.asarray(timesteps,dtype=np.float64)
 if raw.ndim!=3 or raw.shape[0]<1 or raw.shape[1]!=len(times) or raw.shape[2]!=2:raise ValueError('Expected raw_data [N,T,2]')
 if not np.all(np.isfinite(raw)) or np.any(raw<0):raise ValueError('Raw sensitivity has missing/nonfinite/negative values')
 if np.any(np.all(raw==0,axis=2)):raise ValueError('Zero pair may indicate an upstream skipped calibration error; inspect before importing')
 if not np.all(np.isfinite(times)) or len(np.unique(times))!=len(times):raise ValueError('Invalid time grid')
 return {'timesteps':times.tolist(),'J_x_norm':raw[:,:,0].mean(axis=0).tolist(),
  'J_t_norm':raw[:,:,1].mean(axis=0).tolist(),'raw_data':raw.tolist(),
  'aggregation':'arithmetic mean across videos; an explicit project convention because upstream converter is absent'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--method',choices=['magcache','sencache'],required=True)
 p.add_argument('--config',required=True,type=Path,help='One manifest row, or JSON containing generation defaults under generation')
 p.add_argument('--samples',required=True,type=Path,help='JSONL prompt_id,prompt,seed,split=calibration; Sen also video_path')
 p.add_argument('--output',required=True,type=Path);p.add_argument('--vendor-dir',type=Path,default=Path('vendor'))
 p.add_argument('--checkpoint-dir',type=Path);p.add_argument('--device',type=int,default=0)
 p.add_argument('--import-raw',type=Path,help='Convert upstream Sen npz raw_data+timesteps without running GPU')
 p.add_argument('--checkpoint-fingerprint',help='Required when importing existing raw Sen data; must be verified exact weights SHA')
 p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 config=json.loads(a.config.read_text());row=config.get('generation',config)
 if 'model_id' not in row and 'model_id' in config:row['model_id']=config['model_id']
 validate_row(dict(row,method='baseline',prompt='calibration validation',prompt_id='validation',run_id='validation',seed=0))
 samples=read_samples(a.samples,a.method)
 if a.output.exists():raise FileExistsError('Refusing to replace calibration profile; choose new output path')
 if a.dry_run:
  print(json.dumps({'status':'planned_only','method':a.method,'samples':len(samples),'config':config_signature(row)},indent=2));return
 start=time.perf_counter()
 result={'schema_version':1,'method':a.method,'config':config_signature(row),'sample_ids':[s['prompt_id'] for s in samples],
  'prompt_sha256':[hashlib.sha256(canonical_prompt(s['prompt']).encode()).hexdigest() for s in samples],
  'samples':samples,'created_by':'cachebench.calibrate','is_model_training':False}
 if a.method=='sencache' and not a.import_raw:
  video_hashes={}
  for sample in samples:
   digest=hashlib.sha256()
   with Path(sample['video_path']).open('rb') as stream:
    while chunk:=stream.read(8*1024*1024):digest.update(chunk)
   video_hashes[sample['prompt_id']]=digest.hexdigest()
  result['sample_video_sha256']=video_hashes
 if a.import_raw:
  if a.method!='sencache' or not a.checkpoint_fingerprint:p.error('--import-raw requires --method sencache and verified --checkpoint-fingerprint')
  import numpy as np
  with np.load(a.import_raw,allow_pickle=False) as z:result.update(aggregate_sen(z['raw_data'],z['timesteps']))
  if len(result['raw_data'])!=len(samples):raise ValueError('Declared calibration videos must match raw samples count')
  result['checkpoint_fingerprint']=a.checkpoint_fingerprint
  result['imported_raw_sha256']=hashlib.sha256(a.import_raw.read_bytes()).hexdigest()
  result['source_revisions']='user-verified external raw provenance; inspect separately'
 else:
  if not a.checkpoint_dir:p.error('--checkpoint-dir required')
  from .wan_adapter import NativeWanBackend
  backend=NativeWanBackend(a.vendor_dir,a.checkpoint_dir,a.device)
  result.update(run_mag(backend,row,samples) if a.method=='magcache' else run_sen(backend,row,samples))
  backend.sync();result['checkpoint_fingerprint']=backend.fingerprint;result['source_revisions']=backend.revisions
 result['calibration_wall_seconds']=time.perf_counter()-start
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
 print(json.dumps({'status':'complete','output':str(a.output),'calibration_wall_seconds':result['calibration_wall_seconds']},indent=2))
if __name__=='__main__':main()
