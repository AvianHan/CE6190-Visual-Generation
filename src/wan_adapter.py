"""SeaCache's T2V denoising loop with interchangeable official cache forwards.

This is a source integration, not an independently validated reproduction. The
function bodies are selected by AST from pinned upstream source and executed in
an explicit namespace; unrelated upstream CLI/global code is never executed.
The denoising loop comes from SeaCache; model loading, text/noise preparation,
cache reset, timing and optional decoding are shared project integration code.
"""
from __future__ import annotations
import ast, hashlib, importlib.util, json, logging, math, os, sys, tempfile, time, types
from collections import Counter
from pathlib import Path
from .sources import FORWARDS, verify
from .policies import config_signature,canonical_prompt
from .seacache_base import SEACACHE_T2V_FILE,load_seacache_denoising_loop,ObservedScheduler

TEA_COEFFICIENTS={
 False:[2.39676752e3,-1.31110545e3,2.01331979e2,-8.29855975,1.37887774e-1],
 True:[-5.21862437e4,9.23041404e3,-5.28275948e2,1.36987616e1,-4.99875664e-2],
}

def function_source(path, name):
 source=Path(path).read_text()
 tree=ast.parse(source)
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
 if len(nodes)!=1: raise ValueError(f'{path}: expected one {name}')
 return ast.Module(body=nodes,type_ignores=[]), hashlib.sha256(source.encode()).hexdigest()

def extract_function(path,name,namespace):
 tree,digest=function_source(path,name)
 exec(compile(tree,str(path),'exec'),namespace)
 return namespace[name],digest

def audit_shared_residual_contract(wan_path, cache_path, method):
 """Fail closed if a pinned T2V forward changes the shared compute or HOW.

 This is a structural check of the known source integration, not a general
 Python verifier or a substitute for tensor/GPU parity tests. Source revision
 verification remains mandatory. WHEN scores, thresholds, retention rules and
 CFG decision coupling are deliberately left unchanged.
 """
 if method not in FORWARDS:raise ValueError('Unknown cache method')
 def key(node):return ast.dump(node,include_attributes=False)
 def statement(text):return ast.parse(text).body[0]
 def one(values,label):
  if len(values)!=1:raise ValueError(f'{method}: shared residual contract needs one {label}')
  return values[0]
 def assigns_name(node,name):
  return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
 def at(body,name,fragment):
  return one([i for i,n in enumerate(body) if assigns_name(n,name) and fragment in ast.unparse(n)],fragment)
 def forward(path,owner=None):
  body=ast.parse(Path(path).read_text()).body
  if owner:
   body=one([n for n in body if isinstance(n,ast.ClassDef) and n.name==owner],owner).body
  name='forward' if owner else method+'_forward'
  return one([n for n in body if isinstance(n,ast.FunctionDef) and n.name==name],name)
 base=forward(wan_path,'WanModel');cache=forward(cache_path)
 if key(base.args)!=key(cache.args):raise ValueError(f'{method}: forward arguments differ from Wan')
 shared=[]
 for body in (base.body,cache.body):
  start=at(body,'x','self.patch_embedding')
  end=at(body,'kwargs','dict(')
  device=at(body,'device','self.patch_embedding.weight.device')
  # Device placement and optional y concatenation, then patch/time/text prep.
  shared.append(body[device:device+3]+body[start:end+1])
 if [key(n) for n in shared[0]]!=[key(n) for n in shared[1]]:
  raise ValueError(f'{method}: common embeddings/context preparation differs from Wan')
 output=[]
 for body in (base.body,cache.body):
  head=body[at(body,'x','self.head(')]
  unpatch=body[at(body,'x','self.unpatchify(')]
  result=one([n for n in body if isinstance(n,ast.Return)],'output return')
  output.append([head,unpatch,result])
 if [key(n) for n in output[0]]!=[key(n) for n in output[1]]:
  raise ValueError(f'{method}: head/unpatchify/output differs from Wan')
 block_body=one([n for n in base.body if isinstance(n,ast.For)],'Wan block loop').body
 loops=[n for n in ast.walk(cache) if isinstance(n,ast.For)]
 expected_loops=3 if method in {'teacache','seacache'} else 1
 if len(loops)!=expected_loops:raise ValueError(f'{method}: unexpected full-block loop count')
 for loop in loops:
  direct=key(loop.iter)==key(ast.parse('self.blocks',mode='eval').body) and key(loop.target)==key(statement('for block in self.blocks:\n pass').target)
  numbered=(key(loop.iter)==key(ast.parse('enumerate(self.blocks)',mode='eval').body)
            and isinstance(loop.target,ast.Tuple) and len(loop.target.elts)==2
            and all(isinstance(n,ast.Name) for n in loop.target.elts)
            and loop.target.elts[1].id=='block')
  if not (direct or numbered) or [key(n) for n in loop.body]!=[key(n) for n in block_body] or loop.orelse:
   raise ValueError(f'{method}: full-block computation differs from Wan')
 if method in {'teacache','seacache'}:
  how=['ori_x = x.clone()']*2
  for branch in ('even','odd'):
   how += [f'x += self.previous_residual_{branch}',f'self.previous_residual_{branch} = x - ori_x']
 elif method=='magcache':
  how=['ori_x = x','residual_x = self.residual_cache[self.cnt % 2]',
       'x = x + residual_x','residual_x = x - ori_x',
       'self.residual_cache[self.cnt % 2] = residual_x']
 else:
  how=['cached_residual = self.cached_residual_cond if is_cond else self.cached_residual_uncond',
       'x = x + cached_residual','ori_x_patched = x.clone()',
       'new_residual = x - ori_x_patched',
       'self.cached_residual_cond = new_residual.detach()',
       'self.cached_residual_uncond = new_residual.detach()',
       'self.cached_residual_cond = None','self.cached_residual_uncond = None']
 expected_how=[statement(s) for s in how]
 # Every write to x must be an unchanged Wan operation or plain residual add.
 base_x={key(n) for n in ast.walk(base) if assigns_name(n,'x')}
 allowed_x=base_x|{key(n) for n in expected_how if assigns_name(n,'x') or isinstance(n,ast.AugAssign)}
 residual_names={'ori_x','ori_x_patched','residual_x','cached_residual','new_residual',
                 'previous_residual_even','previous_residual_odd','residual_cache',
                 'cached_residual_cond','cached_residual_uncond'}
 def target_name(target):
  if isinstance(target,ast.Subscript):return target_name(target.value)
  if isinstance(target,ast.Attribute):return target.attr
  if isinstance(target,ast.Name):return target.id
  return None
 observed_how=[]
 for node in ast.walk(cache):
  if not isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):continue
  targets=node.targets if isinstance(node,ast.Assign) else [node.target]
  names={target_name(t) for t in targets}
  if 'x' in names and key(node) not in allowed_x:
   raise ValueError(f'{method}: x write violates plain residual/common compute contract')
  if names & residual_names or (isinstance(node,ast.AugAssign) and 'x' in names):
   observed_how.append(node)
  elif assigns_name(node,'x') and key(node) not in base_x:
   observed_how.append(node)
 if Counter(key(n) for n in observed_how)!=Counter(key(n) for n in expected_how):
  raise ValueError(f'{method}: residual storage/reuse differs from plain h_current + cached_r')
 def digest(nodes):return hashlib.sha256('\n'.join(key(n) for n in nodes).encode()).hexdigest()
 return {'wan_forward_ast':digest([base]),
         f'{method}_shared_compute_ast':digest(shared[0]+block_body+output[0]),
         f'{method}_plain_residual_ast':digest(expected_how)}

def read_calibration(path,row,checkpoint_fingerprint,method):
 if not path: raise ValueError(f'{method} requires a calibration file from cachebench.calibrate')
 data=json.loads(Path(path).read_text())
 if data.get('schema_version')!=1 or data.get('method')!=method: raise ValueError('Calibration method/schema mismatch')
 if data.get('config')!=config_signature(row):
  raise ValueError('Calibration does not match checkpoint alias, resolution, frames, steps, CFG, shift, dtype, solver')
 if data.get('checkpoint_fingerprint')!=checkpoint_fingerprint: raise ValueError('Calibration checkpoint content fingerprint mismatch')
 if (row.get('prompt_id') in data.get('sample_ids',[]) or hashlib.sha256(canonical_prompt(row.get('prompt','')).encode()).hexdigest() in data.get('prompt_sha256',[])): raise ValueError('Calibration/evaluation prompt overlap')
 return data


def sea_filter_hook(original,variant):
 """Explicit experimental hooks; never alter an official source function body."""
 if variant=='official':return original
 if variant=='raw_proxy':
  def identity(x,*args,**kwargs):return x
  return identity
 if variant=='no_gain_normalization':
  def no_gain(*args,**kwargs):
   kwargs['norm_mode']='none'
   return original(*args,**kwargs)
  return no_gain
 raise ValueError('Invalid SeaCache variant')

def sensitivity_ablation(jx,jt,variant):
 if variant=='official':return jx,jt
 if variant=='latent_only':return jx,jt*0
 if variant=='time_only':return jx*0,jt
 raise ValueError('Invalid SenCache variant')

class Trace:
 def __init__(self,model):
  self.model=model;self.events=[];self.blocks=0
  self.hook=model.blocks[0].register_forward_pre_hook(self._block)
 def _block(self,*args): self.blocks+=1
 def wrap(self,forward):
  trace=self
  def wrapped(model,*args,**kwargs):
   before=trace.blocks;call=len(trace.events)
   value=forward(*args,**kwargs)
   trace.events.append({'step':call//2,'branch':'conditional' if call%2==0 else 'unconditional',
     'refresh':trace.blocks>before})
   return value
  return types.MethodType(wrapped,self.model)
 def close(self): self.hook.remove()

class NativeWanBackend:
 def __init__(self,vendor,checkpoint,device=0):
  import torch
  self.torch=torch;self.vendor=Path(vendor).resolve();self.checkpoint=Path(checkpoint).resolve()
  self.revisions=verify(self.vendor)
  self.denoising_loop,base_hashes=load_seacache_denoising_loop(
   self.vendor/'seacache'/SEACACHE_T2V_FILE,torch)
  sys.path.insert(0,str(self.vendor/'wan'))
  import wan
  from wan.configs import WAN_CONFIGS
  import wan.modules.attention as attn
  if not attn.FLASH_ATTN_2_AVAILABLE:raise RuntimeError('Pinned backend requires flash-attn2; install requirements documented in setup')
  attn.FLASH_ATTN_3_AVAILABLE=False
  if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for Wan generation; CPU tests are separate')
  torch.cuda.set_device(device)
  torch.backends.cuda.matmul.allow_tf32=False
  torch.backends.cudnn.allow_tf32=False
  self.pipe=wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(self.checkpoint),device_id=device,rank=0,t5_cpu=False)
  if self.pipe.model.dim!=1536 or len(self.pipe.model.blocks)!=30:raise ValueError('Checkpoint is not the supported Wan2.1-T2V-1.3B architecture')
  self.original_forward=self.pipe.model.forward
  self.code_hashes=base_hashes;self.trace=None
  self.fingerprint=self._fingerprint()
 def _fingerprint(self):
  """Content hash of all model/tokenizer files; metadata cache avoids repeated reads.
  Cache is an optimization trusted under unchanged sizes/mtime_ns. Delete the
  fingerprint cache to force a fresh complete content scan.
  """
  files=sorted(p for p in self.checkpoint.rglob('*') if p.is_file() and '.cache' not in p.parts and '.git' not in p.parts)
  if not files: raise ValueError('Empty checkpoint')
  signature=[(str(p.relative_to(self.checkpoint)),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
  key=hashlib.sha256(str(self.checkpoint).encode()).hexdigest()
  cache=Path.home()/'.cache'/'ce6190-cachebench'/f'{key}.json';cache.parent.mkdir(parents=True,exist_ok=True)
  if cache.exists():
   old=json.loads(cache.read_text())
   if old.get('stat_signature')==[list(s) for s in signature]: return old['fingerprint']
  digest=hashlib.sha256()
  for p in files:
   digest.update(str(p.relative_to(self.checkpoint)).encode());digest.update(b'\0')
   with p.open('rb') as f:
    while chunk:=f.read(8*1024*1024):digest.update(chunk)
  fingerprint=digest.hexdigest()
  fd,tmp=tempfile.mkstemp(prefix=cache.stem+'-',suffix='.tmp',dir=cache.parent)
  try:
   with os.fdopen(fd,'w') as stream:json.dump({'stat_signature':signature,'fingerprint':fingerprint},stream)
   os.replace(tmp,cache)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
  return fingerprint
 def make_scheduler(self,row):
  from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler,get_sampling_sigmas,retrieve_timesteps
  from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
  common={'num_train_timesteps':self.pipe.num_train_timesteps,'shift':1,'use_dynamic_shifting':False}
  if row.get('sample_solver','unipc')=='unipc':
   scheduler=FlowUniPCMultistepScheduler(**common)
   scheduler.set_timesteps(row['num_inference_steps'],device=self.pipe.device,shift=row['flow_shift'])
  else:
   scheduler=FlowDPMSolverMultistepScheduler(**common)
   retrieve_timesteps(scheduler,device=self.pipe.device,sigmas=get_sampling_sigmas(row['num_inference_steps'],row['flow_shift']))
  return scheduler
 def shape(self,row):
  p=self.pipe
  shape=(p.vae.model.z_dim,(row['num_frames']-1)//p.vae_stride[0]+1,row['height']//p.vae_stride[1],row['width']//p.vae_stride[2])
  seq_len=math.ceil(shape[1]*shape[2]*shape[3]/(p.patch_size[1]*p.patch_size[2]))
  return shape,seq_len
 def encode_text(self,prompt,negative=None):
  # Preserve this project's explicit empty negative prompt. SeaCache's CLI
  # instead substitutes the Wan default when n_prompt is an empty string.
  p=self.pipe;p.text_encoder.model.to(p.device)
  context=p.text_encoder([prompt],p.device)
  context_null=p.text_encoder([negative if negative is not None else p.sample_neg_prompt],p.device)
  return context,context_null
 def _official_function(self,method,variant='official'):
  import numpy as np
  import torch.cuda.amp as amp
  from wan.modules.model import sinusoidal_embedding_1d
  namespace={'torch':self.torch,'amp':amp,'np':np,'logging':logging,'sinusoidal_embedding_1d':sinusoidal_embedding_1d}
  if method=='seacache':
   path=self.vendor/'seacache'/'Wan2.1'/'util_seacache.py'
   spec=importlib.util.spec_from_file_location('cachebench_official_sea_util',path)
   mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
   namespace.update(apply_sea_with_scheduler=sea_filter_hook(mod.apply_sea_with_scheduler,variant),rel_l1=mod.rel_l1)
   self.code_hashes['sea_util']=hashlib.sha256(path.read_bytes()).hexdigest()
  path,name=FORWARDS[method]
  self.code_hashes.update(audit_shared_residual_contract(
   self.vendor/'wan'/'wan'/'modules'/'model.py',self.vendor/method/path,method))
  fn,digest=extract_function(self.vendor/method/path,name,namespace)
  self.code_hashes[method]=digest
  return fn
 def patch(self,row,scheduler):
  import numpy as np
  method=row['method'];m=self.pipe.model
  m.forward=self.original_forward
  if self.trace is not None:self.trace.close()
  if method!='baseline':
   fn=self._official_function(method,row.get('variant','official'))
   m.forward=types.MethodType(fn,m)
   m.cnt=0;m.num_steps=row['num_inference_steps']*2
   if method in {'teacache','seacache'}:
    setattr(m,f'enable_{method}',True);setattr(m,f'{method}_thresh',row['threshold'])
    for branch in ('even','odd'):
     setattr(m,f'accumulated_rel_l1_distance_{branch}',0.0)
     setattr(m,f'previous_e0_{branch}',None);setattr(m,f'previous_residual_{branch}',None)
    retention=bool(row.get('use_ret_steps',False))
    m.ret_steps=10 if retention else 2;m.cutoff_steps=m.num_steps if retention else m.num_steps-2
    if method=='teacache':m.use_ref_steps=retention;m.coefficients=TEA_COEFFICIENTS[retention]
    else:m.scheduler=scheduler
   elif method=='magcache':
    data=read_calibration(row.get('calibration_path'),row,self.fingerprint,method)
    if data.get('source_revisions')!=self.revisions and not row.get('allow_external_calibration',False):raise ValueError('Calibration source revisions mismatch or unknown; rerun or explicitly permit external profile')
    ratios=data['mag_ratios']
    if len(ratios)!=m.num_steps or any(not math.isfinite(x) or x<=0 for x in ratios):raise ValueError('Bad MagCache ratio profile')
    m.mag_ratios=np.asarray(ratios);m.K=int(row.get('max_skip',4));m.retention_ratio=float(row.get('retention_ratio',0.2))
    if int(m.num_steps*m.retention_ratio)<2:raise ValueError('MagCache must compute both CFG branches before reuse')
    m.magcache_thresh=row['threshold'];m.residual_cache=[None,None]
    m.accumulated_ratio=[1.0,1.0];m.accumulated_err=[0.0,0.0];m.accumulated_steps=[0,0]
   elif method=='sencache':
    data=read_calibration(row.get('calibration_path'),row,self.fingerprint,method)
    if data.get('source_revisions')!=self.revisions and not row.get('allow_external_calibration',False):raise ValueError('Calibration source revisions mismatch or unknown; rerun or explicitly permit external profile')
    times=np.asarray(data['timesteps']);jx=np.asarray(data['J_x_norm']);jt=np.asarray(data['J_t_norm'])
    if not(len(times)==len(jx)==len(jt)==row['num_inference_steps']):raise ValueError('Bad SenCache calibration dimensions')
    if not np.all(np.isfinite(jx)) or not np.all(np.isfinite(jt)) or np.any(jx<0) or np.any(jt<0):raise ValueError('Invalid sensitivities')
    if not np.all(np.isfinite(times)) or len(np.unique(times))!=len(times):raise ValueError('Invalid sensitivity timesteps')
    actual=scheduler.timesteps.detach().float().cpu().numpy()
    if not np.allclose(times,actual,rtol=1e-6,atol=1e-4):raise ValueError('Sensitivity timestep grid mismatch')
    jx,jt=sensitivity_ablation(jx,jt,row.get('variant','official'))
    m.J_z_norm=jx.tolist();m.J_t_norm=jt.tolist();m.sencache_timesteps_array=times
    scale_mode=row.get('sencache_scale','official_81f')
    if scale_mode not in {'latent_rms','official_81f'}:raise ValueError('Unknown SenCache scale')
    if scale_mode=='official_81f' and (row['num_frames'],row['height'],row['width'])!=(81,480,832):raise ValueError('official_81f scale requires81frames480x832; use explicit latent_rms variant for changed geometry')
    scale=math.sqrt(math.prod(self.shape(row)[0])) if scale_mode=='latent_rms' else 1447.97790038
    m.sencache_threshold_start=float(row.get('threshold_start',0.045))*scale
    m.sencache_threshold_main=row['threshold']*scale;m.sencache_K=int(row.get('max_skip',3))
    m.cached_z=m.cached_t=m.cached_residual_cond=m.cached_residual_uncond=None
    m.cached_J_z_norm=m.cached_J_t_norm=None;m.accumulated_skips=0;m.skip_this_step=False
    m.current_skip_count=0;m.all_skip_counts=[];m.threshold_switch_step=int(round(m.num_steps*0.2))
    m.ret_steps=0;m.cutoff_steps=m.num_steps
  self.trace=Trace(m);m.forward=self.trace.wrap(m.forward)
 def sync(self):self.torch.cuda.synchronize(self.pipe.device)
 def generate(self,row,decode=True,observer=None):
  torch=self.torch;p=self.pipe
  scheduler=self.make_scheduler(row);self.patch(row,scheduler)
  shape,seq_len=self.shape(row)
  seed_g=torch.Generator(device=p.device).manual_seed(row['seed'])
  self.sync();start=time.perf_counter()
  with torch.no_grad():context,context_null=self.encode_text(row['prompt'],row.get('negative_prompt'))
  self.sync();text_done=time.perf_counter()
  latent=torch.randn(shape,dtype=torch.float32,device=p.device,generator=seed_g)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   loop_scheduler=ObservedScheduler(scheduler,observer) if observer is not None else scheduler
   latents=self.denoising_loop(
    p,[latent],scheduler.timesteps,
    {'context':context,'seq_len':seq_len},{'context':context_null,'seq_len':seq_len},
    row['guidance_scale'],loop_scheduler,seed_g)
   latent=latents[0]
   self.sync();denoise_done=time.perf_counter()
   video=p.vae.decode([latent])[0] if decode else None
   self.sync();end=time.perf_counter()
  return video,{'generation_seconds':end-start,'text_encode_seconds':text_done-start,
    'denoise_seconds':denoise_done-text_done,'vae_decode_seconds':end-denoise_done},list(self.trace.events)
