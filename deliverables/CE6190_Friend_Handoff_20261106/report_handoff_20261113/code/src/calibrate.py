"""Fresh common-configuration calibration and conversion of official Sen raw data.

MagCache: mean of official token-wise residual magnitude ratios on independent
full-compute generated prompts. SenCache: official finite-difference functions on
independent real videos, then explicit arithmetic mean over videos. This supplies
the converter missing from the upstream script; it is an experimental convention,
not a claim that the authors' released profile used this exact aggregation.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, tempfile, time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .policies import config_signature,validate_row,canonical_prompt

def read_samples(path,method):
 if method not in {'magcache','sencache'}:raise ValueError('Unknown calibration method')
 path=Path(path).expanduser().resolve()
 samples=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 if not samples:raise ValueError('Empty calibration set')
 ids=set();prompts=set();videos=set()
 for s in samples:
  if not isinstance(s,dict):raise ValueError('Every calibration sample must be a JSON object')
  if s.get('split')!='calibration':raise ValueError('Every row must explicitly have split=calibration')
  if not isinstance(s.get('prompt_id'),str) or not s['prompt_id'].strip() or not isinstance(s.get('prompt'),str) or not s['prompt'].strip():raise ValueError('Every calibration sample needs prompt_id and prompt')
  if s['prompt_id'] in ids or canonical_prompt(s['prompt']) in prompts:raise ValueError('Duplicate calibration sample')
  ids.add(s['prompt_id']);prompts.add(canonical_prompt(s['prompt']))
  seed=s.get('seed',0)
  if type(seed) is not int or seed<0:raise ValueError('Calibration seed must be a nonnegative integer')
  if method=='sencache':
   if not isinstance(s.get('video_path'),str) or not s['video_path'].strip():raise ValueError('SenCache needs actual independent calibration videos')
   video=Path(s['video_path']).expanduser()
   video=(video if video.is_absolute() else path.parent/video).resolve()
   if str(video) in videos:raise ValueError('Duplicate calibration video path')
   videos.add(str(video));s['video_path']=str(video)
 return samples

def validate_exclusions(samples,paths):
 """Check the entire calibration set, even when this task handles one sample."""
 from .manifest import load_prompts
 ids={s['prompt_id'] for s in samples};prompts={canonical_prompt(s['prompt']) for s in samples}
 records=[]
 for path in paths:
  path=Path(path).expanduser().resolve()
  rows=load_prompts(path)
  for row in rows:
   if row['prompt_id'] in ids or canonical_prompt(row['prompt']) in prompts:
    raise ValueError(f'Calibration/evaluation overlap in {path}: {row["prompt_id"]}')
  records.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prompts':len(rows)})
 return records

def sha256_file(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while chunk:=stream.read(8*1024*1024):digest.update(chunk)
 return digest.hexdigest()

def _valid_sha256(value):
 return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def fetch_calibration_video(sample):
 """Fetch an explicitly supplied public HTTPS asset; never guess a media URL.

 Existing files are reused and hashed. A supplied video_sha256 is enforced;
 otherwise the observed hash is recorded, without claiming prior verification.
 The destination is installed atomically and an existing file is never replaced.
 """
 path=Path(sample['video_path']).expanduser().resolve()
 expected=sample.get('video_sha256')
 if expected is not None and not _valid_sha256(expected):raise ValueError('video_sha256 must be a lowercase SHA-256 digest')
 url=sample.get('video_url')
 def record(status):
  if path.stat().st_size==0:raise ValueError(f'Empty calibration video: {path}')
  actual=sha256_file(path)
  if expected is not None and actual!=expected:raise ValueError(f'Calibration video checksum mismatch: {path}')
  return {'path':str(path),'sha256':actual,'expected_sha256':expected,
          'checksum_status':'matches_expected' if expected is not None else 'observed_only',
          'source_url':url,'status':status,'bytes':path.stat().st_size}
 if path.is_file():return record('existing')
 if not isinstance(url,str) or urlparse(url).scheme!='https' or not urlparse(url).netloc:
  raise ValueError(f'Missing local video {path}; provide an explicit HTTPS video_url')
 path.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(3):
  fd,temporary=tempfile.mkstemp(prefix=path.name+'.download-',dir=path.parent)
  try:
   with os.fdopen(fd,'wb') as target:
    with urlopen(Request(url,headers={'User-Agent':'CE6190-calibration/1.0'}),timeout=30) as response:
     if urlparse(response.geturl()).scheme!='https':raise ValueError('Video download redirected away from HTTPS')
     content_type=response.headers.get('Content-Type','').lower()
     if 'text/html' in content_type or 'application/json' in content_type:raise ValueError('Video URL returned a page instead of video bytes')
     digest=hashlib.sha256();size=0
     while chunk:=response.read(1024*1024):
      target.write(chunk);digest.update(chunk);size+=len(chunk)
     if not size:raise ValueError('Empty calibration video download')
     if expected is not None and digest.hexdigest()!=expected:raise ValueError('Downloaded calibration video checksum mismatch')
    target.flush();os.fsync(target.fileno())
   try:os.link(temporary,path)
   except FileExistsError:return record('existing')
   return record('downloaded')
  except (OSError,ValueError):
   if attempt==2:raise
   time.sleep(2**attempt)
  finally:
   if os.path.exists(temporary):os.unlink(temporary)

def write_profile_atomic(path,data):
 """Publish a complete profile without clobbering any earlier result."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as stream:
   json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  os.link(temporary,path)
 finally:
  if os.path.exists(temporary):os.unlink(temporary)

def _profile_header(method,row,samples,indices,manifest_path,exclusions):
 selected=[samples[i] for i in indices]
 return {'schema_version':1,'method':method,'config':config_signature(row),
  'sample_ids':[s['prompt_id'] for s in selected],
  'prompt_sha256':[hashlib.sha256(canonical_prompt(s['prompt']).encode()).hexdigest() for s in selected],
  'samples':selected,'created_by':'cachebench.calibrate','is_model_training':False,
  'selection':{'manifest_path':str(Path(manifest_path).resolve()),
   'manifest_sha256':sha256_file(manifest_path),'total_samples':len(samples),'sample_indices':list(indices)},
  'excluded_prompt_files':exclusions}

def _number(value,*,positive=False):
 if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (positive and value<=0):
  raise ValueError('Calibration contains invalid numeric values')
 return float(value)

def _mean_columns(rows):
 return [math.fsum(row[j] for row in rows)/len(rows) for j in range(len(rows[0]))]

def _check_vector(actual,expected,label):
 if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError(f'Bad {label} dimensions')
 if any(not math.isclose(_number(a),b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(actual,expected)):
  raise ValueError(f'{label} differs from retained raw calibration values')

def merge_profiles(paths,method,row,samples,manifest_path,exclusions):
 """Merge complete sample coverage using stdlib only; no GPU/model imports.

 Every raw value is retained in original manifest order. The sum of shard
 durations measures aggregate work, not elapsed parallel wall time, so the
 merged calibration_wall_seconds is explicitly unknown (None).
 """
 started=time.perf_counter()
 if not paths:raise ValueError('No calibration shards supplied')
 if method not in {'magcache','sencache'}:raise ValueError('Unknown calibration method')
 expected=_profile_header(method,row,samples,range(len(samples)),manifest_path,exclusions)
 count=len(samples);steps=row['num_inference_steps'];ordered={};records=[];common=None;times=None;video_hashes={};downloads={}
 for path in paths:
  path=Path(path).expanduser().resolve();data=json.loads(path.read_text())
  for key in ('schema_version','method','config','created_by','is_model_training','excluded_prompt_files'):
   if data.get(key)!=expected[key]:raise ValueError(f'{path}: incompatible {key}')
  selection=data.get('selection',{})
  for key in ('manifest_sha256','total_samples'):
   if selection.get(key)!=expected['selection'][key]:raise ValueError(f'{path}: calibration selection mismatch')
  indices=selection.get('sample_indices')
  if not isinstance(indices,list) or not indices or any(type(i) is not int or not 0<=i<count for i in indices) or len(set(indices))!=len(indices):
   raise ValueError(f'{path}: invalid sample indices')
  header=_profile_header(method,row,samples,indices,manifest_path,exclusions)
  for key in ('samples','sample_ids','prompt_sha256'):
   if data.get(key)!=header[key]:raise ValueError(f'{path}: sample declaration mismatch')
  if not _valid_sha256(data.get('checkpoint_fingerprint')):raise ValueError(f'{path}: invalid checkpoint fingerprint')
  revisions=data.get('source_revisions')
  if not isinstance(revisions,dict) or not revisions:raise ValueError(f'{path}: missing verified source revisions')
  if not isinstance(data.get('aggregation'),str) or not data['aggregation']:raise ValueError(f'{path}: missing aggregation convention')
  if method=='sencache' and not _valid_sha256(data.get('calibration_code_sha256')):raise ValueError(f'{path}: missing calibration code checksum')
  provenance={k:data.get(k) for k in ('checkpoint_fingerprint','source_revisions','aggregation','calibration_code_sha256','geometry_note','estimator_note')}
  if common is None:common=provenance
  elif common!=provenance:raise ValueError(f'{path}: calibration provenance differs across shards')
  wall=_number(data.get('calibration_wall_seconds'))
  if method=='magcache':
   raw=data.get('raw_ratios')
   if not isinstance(raw,list) or len(raw)!=len(indices):raise ValueError(f'{path}: raw ratio sample count mismatch')
   if any(not isinstance(v,list) or len(v)!=2*steps for v in raw):raise ValueError(f'{path}: raw ratio shape mismatch')
   raw=[[_number(v,positive=True) for v in values] for values in raw]
   _check_vector(data.get('mag_ratios'),_mean_columns(raw),'mag_ratios')
  else:
   actual_times=data.get('timesteps')
   if not isinstance(actual_times,list) or len(actual_times)!=steps:raise ValueError(f'{path}: timestep shape mismatch')
   actual_times=[_number(v) for v in actual_times]
   if len(set(actual_times))!=steps:raise ValueError(f'{path}: duplicate timesteps')
   if times is None:times=actual_times
   elif times!=actual_times:raise ValueError(f'{path}: timestep grid differs across shards')
   raw=data.get('raw_data')
   if not isinstance(raw,list) or len(raw)!=len(indices):raise ValueError(f'{path}: raw sensitivity sample count mismatch')
   if any(not isinstance(v,list) or len(v)!=steps for v in raw):raise ValueError(f'{path}: raw sensitivity shape mismatch')
   clean=[]
   for values in raw:
    pairs=[]
    for pair in values:
     if not isinstance(pair,list) or len(pair)!=2:raise ValueError(f'{path}: raw sensitivity pair mismatch')
     pair=[_number(v) for v in pair]
     if pair==[0.0,0.0]:raise ValueError(f'{path}: zero sensitivity pair may be a failed calibration step')
     pairs.append(pair)
    clean.append(pairs)
   raw=clean
   for dimension,key in enumerate(('J_x_norm','J_t_norm')):
    means=_mean_columns([[pair[dimension] for pair in values] for values in raw])
    _check_vector(data.get(key),means,key)
   hashes=data.get('sample_video_sha256',{})
   if not isinstance(hashes,dict) or set(hashes)!=set(header['sample_ids']) or any(not _valid_sha256(v) for v in hashes.values()):
    raise ValueError(f'{path}: missing video checksums')
   for sample in header['samples']:
    supplied=sample.get('video_sha256')
    if supplied is not None and supplied!=hashes[sample['prompt_id']]:raise ValueError(f'{path}: video checksum disagrees with manifest')
   video_hashes.update(hashes)
   downloaded=data.get('video_downloads',{})
   if not isinstance(downloaded,dict) or set(downloaded)-set(header['sample_ids']):raise ValueError(f'{path}: invalid download provenance')
   downloads.update(downloaded)
  for index,values in zip(indices,raw):
   if index in ordered:raise ValueError(f'Duplicate calibration sample index {index}')
   ordered[index]=values
  records.append({'path':str(path),'sha256':sha256_file(path),'sample_indices':indices,'calibration_wall_seconds':wall})
 if set(ordered)!=set(range(count)):raise ValueError(f'Missing calibration sample indices: {sorted(set(range(count))-set(ordered))}')
 raw=[ordered[i] for i in range(count)]
 result=expected|{k:v for k,v in common.items() if v is not None}
 if method=='magcache':result.update(raw_ratios=raw,mag_ratios=_mean_columns(raw))
 else:
  if len(set(video_hashes.values()))!=count:raise ValueError('Different calibration samples contain identical video files')
  result.update(raw_data=raw,timesteps=times,
   J_x_norm=_mean_columns([[pair[0] for pair in values] for values in raw]),
   J_t_norm=_mean_columns([[pair[1] for pair in values] for values in raw]),
   sample_video_sha256=video_hashes,video_downloads=downloads)
 result.update(calibration_wall_seconds=None,profile_kind='merged',shards=records,
  calibration_time_accounting={'note':'Sum includes each shard preparation/model load; parallel elapsed wall time is not measured here.',
   'sum_shard_wall_seconds':math.fsum(r['calibration_wall_seconds'] for r in records),
   'max_shard_wall_seconds':max(r['calibration_wall_seconds'] for r in records),
   'merge_wall_seconds':time.perf_counter()-started,'shard_count':len(records)})
 return result

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

def selected_video_frames(reader,total_frames,indices):
 """Yield requested frames while decoding sequentially, with bounded memory.

 This helper intentionally uses no NumPy/Torch so the decoding/count contract
 can be checked in the offline launcher tests. Duplicated sample indices are
 expanded by the caller after resizing each distinct source frame once.
 """
 if type(total_frames) is not int or total_frames<1:raise ValueError('Video contains no frames or an invalid frame count')
 wanted=set(indices)
 if not wanted or any(type(i) is not int or not 0<=i<total_frames for i in wanted):raise ValueError('Invalid sampled frame indices')
 observed=0
 for index,frame in enumerate(reader):
  observed=index+1
  if observed>total_frames:raise ValueError('Decoded video contains more frames than count_frames reported')
  if index in wanted:yield index,frame
 if observed!=total_frames:raise ValueError(f'Decoded frame count {observed} differs from count_frames {total_frames}')

def load_video(path,row,device):
 """Uniform sampling with the original float32 GPU resize/normalization.

 Decode one frame at a time instead of holding the full-resolution video or
 all sampled full-resolution frames in RAM. Only the target-sized float32
 tensor and one decoded source frame need to be resident.
 """
 import numpy as np
 import torch
 import torch.nn.functional as F
 path=Path(path)
 reader=None
 try:
  if path.suffix.lower()=='.npy':
   frames=np.load(path,allow_pickle=False,mmap_mode='r')
   if frames.ndim!=4 or frames.shape[-1]!=3 or frames.dtype!=np.uint8:raise ValueError(f'{path}: expected uint8 T,H,W,3 RGB')
   total_frames=len(frames)
  else:
   import imageio.v2 as iio
   reader=iio.get_reader(str(path),format='FFMPEG')
   total_frames=int(reader.count_frames())
  if total_frames<1:raise ValueError(f'{path}: video contains no frames')
  if type(row['num_frames']) is not int or row['num_frames']<1:raise ValueError('Requested frame count must be positive')
  indices=np.linspace(0,total_frames-1,row['num_frames'],dtype=int).tolist()
  positions={}
  for output_index,frame_index in enumerate(indices):positions.setdefault(frame_index,[]).append(output_index)
  output=torch.empty((row['num_frames'],3,row['height'],row['width']),device=device,dtype=torch.float32)
  selected=selected_video_frames(reader,total_frames,indices) if reader is not None else ((i,frames[i]) for i in positions)
  for frame_index,frame in selected:
   if frame.ndim!=3 or frame.shape[-1]!=3 or frame.dtype!=np.uint8 or min(frame.shape[:2])<1:
    raise ValueError(f'{path}: decoded frame {frame_index} is not uint8 H,W,3 RGB')
   x=torch.from_numpy(frame.copy()).to(device=device,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
   x=F.interpolate(x,size=(row['height'],row['width']),mode='bilinear',align_corners=False,antialias=True)
   x=x/127.5-1
   for output_index in positions[frame_index]:output[output_index].copy_(x[0])
  return output.permute(1,0,2,3)
 finally:
  if reader is not None:reader.close()

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
   'geometry_note':'Uniform np.linspace frame indices; bounded sequential video decoding (npy uses mmap); unchanged float32 per-frame GPU bilinear resize with antialias and (height,width), then x/127.5-1. Corrects upstream ambiguous W/H naming.',
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
 operation=p.add_mutually_exclusive_group()
 operation.add_argument('--import-raw',type=Path,help='Convert upstream Sen npz raw_data+timesteps without running GPU')
 operation.add_argument('--sample-index',type=int,help='Process one zero-based sample index, e.g. SLURM_ARRAY_TASK_ID')
 operation.add_argument('--merge-profiles',nargs='+',type=Path,help='Merge all sample shards on CPU; no model loading or GPU imports')
 p.add_argument('--exclude-prompts',action='append',default=[],type=Path,help='Reject overlap with this development/test prompt file; repeat for every held-out set')
 p.add_argument('--checkpoint-fingerprint',help='Required when importing existing raw Sen data; must be verified exact weights SHA')
 p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 config=json.loads(a.config.read_text());row=config.get('generation',config)
 if 'model_id' not in row and 'model_id' in config:row['model_id']=config['model_id']
 validate_row(dict(row,method='baseline',prompt='calibration validation',prompt_id='validation',run_id='validation',seed=0))
 all_samples=read_samples(a.samples,a.method)
 exclusions=validate_exclusions(all_samples,a.exclude_prompts)
 if a.sample_index is not None and not 0<=a.sample_index<len(all_samples):p.error('--sample-index is outside the calibration manifest')
 indices=[a.sample_index] if a.sample_index is not None else list(range(len(all_samples)))
 samples=[all_samples[i] for i in indices]
 if a.output.exists():raise FileExistsError('Refusing to replace calibration profile; choose new output path')
 if a.dry_run:
  if a.merge_profiles:merge_profiles(a.merge_profiles,a.method,row,all_samples,a.samples,exclusions)
  print(json.dumps({'status':'planned_only','method':a.method,'samples':len(samples),
   'total_samples':len(all_samples),'sample_indices':indices,'merge_profiles':a.merge_profiles is not None,
   'excluded_prompt_files':exclusions,'config':config_signature(row)},indent=2));return
 if a.merge_profiles:
  result=merge_profiles(a.merge_profiles,a.method,row,all_samples,a.samples,exclusions)
  write_profile_atomic(a.output,result)
  print(json.dumps({'status':'complete','output':str(a.output),'profile_kind':'merged',
   'samples':len(all_samples),'calibration_time_accounting':result['calibration_time_accounting']},indent=2));return
 start=time.perf_counter()
 result=_profile_header(a.method,row,all_samples,indices,a.samples,exclusions)
 result['profile_kind']='sample_shard' if a.sample_index is not None else 'complete'
 if a.method=='sencache' and not a.import_raw:
  downloads={}
  for sample in samples:
   downloads[sample['prompt_id']]=fetch_calibration_video(sample)
  result['video_downloads']=downloads
  result['sample_video_sha256']={key:value['sha256'] for key,value in downloads.items()}
  if len(set(result['sample_video_sha256'].values()))!=len(samples):raise ValueError('Different calibration samples contain identical video files')
 if a.import_raw:
  if a.method!='sencache' or not _valid_sha256(a.checkpoint_fingerprint):p.error('--import-raw requires --method sencache and verified SHA-256 --checkpoint-fingerprint')
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
 write_profile_atomic(a.output,result)
 print(json.dumps({'status':'complete','output':str(a.output),'calibration_wall_seconds':result['calibration_wall_seconds']},indent=2))
if __name__=='__main__':main()
