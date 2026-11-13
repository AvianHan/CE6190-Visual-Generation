"""Generate one manifest item using the SeaCache-based common Wan loop."""
from __future__ import annotations
import argparse, importlib.metadata, json, os, platform, statistics, sys, time, traceback
from pathlib import Path
from .policies import validate_row

def load_row(path,index):
 rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
 if index<0 or index>=len(rows):raise IndexError('manifest index outside bounds')
 return validate_row(rows[index])

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',required=True,type=Path);p.add_argument('--index',required=True,type=int)
 p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--checkpoint-dir',type=Path)
 p.add_argument('--vendor-dir',type=Path,default=Path('vendor'));p.add_argument('--wan-repo',type=Path,
  help='Optional verification of the vendor/wan path; use sources --directory for setup')
 p.add_argument('--device',type=int,default=0);p.add_argument('--dry-run',action='store_true')
 p.add_argument('--overwrite-failed',action='store_true');a=p.parse_args()
 row=load_row(a.manifest,a.index)
 if a.dry_run:
  print(json.dumps({'status':'planned_only','config':row,'checkpoint':str(a.checkpoint_dir),
    'backend':'seacache_wan_official_loop','requires_cuda':True},indent=2));return
 if not a.checkpoint_dir:p.error('--checkpoint-dir is required for actual generation')
 if a.wan_repo and a.wan_repo.resolve()!=(a.vendor_dir/'wan').resolve():p.error('--wan-repo must equal --vendor-dir/wan')
 out=a.output_dir/row['run_id']
 if out.exists():
  previous=out/'metrics.json'
  if not a.overwrite_failed or not previous.exists() or json.loads(previous.read_text()).get('status')!='failed':
   raise FileExistsError(f'Refusing to overwrite {out}; use a new output root or --overwrite-failed for failed runs')
 out.mkdir(parents=True,exist_ok=True)
 result={'schema_version':1,'status':'running','config':row,'python':sys.version,'platform':platform.platform()}
 (out/'metrics.json').write_text(json.dumps(result,indent=2))
 try:
  import numpy as np
  import torch
  from .wan_adapter import NativeWanBackend
  load_start=time.perf_counter();backend=NativeWanBackend(a.vendor_dir,a.checkpoint_dir,a.device)
  backend.sync();result['model_load_seconds']=time.perf_counter()-load_start
  result['checkpoint_fingerprint']=backend.fingerprint;result['source_revisions']=backend.revisions;result['actual_source_revisions']=backend.revisions
  result['gpu_name']=torch.cuda.get_device_name(a.device)
  result['versions']={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','diffusers','transformers','accelerate','flash-attn','imageio','Pillow')}
  result['cuda_version']=torch.version.cuda
  result['cudnn_version']=torch.backends.cudnn.version()
  result['gpu_capability']=list(torch.cuda.get_device_capability(a.device))
  result['environment']={key:result[key] for key in ('gpu_name','versions','cuda_version','cudnn_version','gpu_capability','python','platform')}
  result['environment']['generation_backend']='seacache_wan_official_loop'
  result['checkpoint_fingerprint_kind']='sha256 of relative filenames and complete file contents; cached on unchanged size/mtime_ns'
  result['source_validation']='pinned clean git commits; unchanged official cache forward bodies and SeaCache t2v_generate denoising loop; project-owned setup and measurement'
  result['implementation_notes']={'variant':row.get('variant','official'),'sencache_scale':row.get('sencache_scale','official_81f'),'teacache_proxy':'e0' if row.get('use_ret_steps',False) else 'e','precision':'bfloat16 autocast with native FP32 operations','attention_backend':'flash_attention_2'}
  warmup=int(row.get('warmup',1));repeats=int(row.get('timing_repeats',3))
  if warmup<0 or repeats<1:raise ValueError('warmup>=0 and timing_repeats>=1 required')
  for _ in range(warmup):
   warm,_,_=backend.generate(row);del warm
  backend.sync();torch.cuda.reset_peak_memory_stats(a.device)
  timings=[];all_trace=[];video=None
  for r in range(repeats):
   if video is not None:del video
   video,timing,trace=backend.generate(row);timings.append(timing);all_trace.append(trace)
  result['peak_memory_gb']=torch.cuda.max_memory_allocated(a.device)/1024**3
  # Decode into canonical uint8 RGB before lossy MP4 encoding. Metrics use frames.npy.
  frames=((video.detach().float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
  np.save(out/'frames.npy',frames,allow_pickle=False)
  import imageio.v2 as imageio
  imageio.mimwrite(str(out/'video.mp4'),frames,fps=float(row.get('fps',16)),codec='libx264',quality=8,macro_block_size=1)
  for key in timings[0]:
   result[key]=statistics.median(t[key] for t in timings)
   result[key+'_repeats']=[t[key] for t in timings]
  if row.get('calibration_path'):
   import hashlib
   cp=Path(row['calibration_path']);profile=json.loads(cp.read_text())
   result['calibration_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
   result['calibration_provenance']={k:profile.get(k) for k in ('sample_ids','config','source_revisions','aggregation','calibration_wall_seconds')}
  result['warmup']=warmup;result['timing_repeats']=repeats
  result['forward_calls']=len(all_trace[-1]);result['refresh_calls']=sum(x['refresh'] for x in all_trace[-1])
  result['refresh_ratio']=result['refresh_calls']/result['forward_calls']
  result['code_sha256']=backend.code_hashes;result['status']='complete'
  result['timing_scope']='text encoding + noise creation + denoising + VAE decode; excludes load, scheduler/policy initialization, profile-file reading, transfer to CPU and video/file encoding'
  result['frames_shape']=list(frames.shape)
  result['backend']='seacache_wan_official_loop'
  (out/'trace.json').write_text(json.dumps({'repeats':all_trace},indent=2))
 except Exception as error:
  result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
  (out/'metrics.json').write_text(json.dumps(result,indent=2));raise
 (out/'metrics.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({'status':'complete','run_id':row['run_id'],'output':str(out)},indent=2))
if __name__=='__main__':main()
