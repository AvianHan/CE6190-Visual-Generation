"""Pure scalar reference rules for tests/teaching; production uses official forwards.
No GPU speed/quality claims are derived from these reference rule tests.
"""
from dataclasses import dataclass
import math

def polynomial(coefficients, x):
 y=0.0
 for c in coefficients: y=y*x+c
 return y

@dataclass
class AccumulatedDistance:
 threshold: float
 accumulated: float=0.0
 def decide(self, delta:float, force=False):
  if not math.isfinite(delta): raise ValueError('nonfinite distance')
  self.accumulated += delta
  refresh=force or self.accumulated>=self.threshold
  if refresh: self.accumulated=0.0
  return refresh

@dataclass
class MagnitudeRule:
 threshold:float
 max_skip:int
 ratio:float=1.0
 error:float=0.0
 steps:int=0
 def decide(self, ratio, force=False):
  if not math.isfinite(ratio) or ratio<=0: raise ValueError('invalid ratio')
  self.ratio*=ratio; self.steps+=1
  self.error+=abs(1-self.ratio)
  refresh=force or not (self.error<self.threshold and self.steps<=self.max_skip)
  if refresh: self.ratio,self.error,self.steps=1.0,0.0,0
  return refresh

def sensitivity_score(jx,jt,delta_x,delta_t):
 values=(jx,jt,delta_x,delta_t)
 if any(not math.isfinite(v) or v<0 for v in values): raise ValueError('invalid sensitivity input')
 return jx*delta_x+jt*delta_t

def canonical_prompt(text):
 return ' '.join(text.split()).casefold()

def config_signature(row):
 """Settings whose change requires fresh calibration/explicit approval."""
 keys=('model_id','height','width','num_frames','num_inference_steps','guidance_scale','flow_shift','dtype','sample_solver','negative_prompt','attention_backend','generation_backend')
 defaults={'sample_solver':'unipc','dtype':'bfloat16','negative_prompt':None,'attention_backend':'flash_attention_2','generation_backend':'seacache_wan'}
 return {k:row.get(k,defaults.get(k)) for k in keys}

def validate_row(row):
 variants={'baseline':{'official'},'teacache':{'official'},'magcache':{'official'},'seacache':{'official','raw_proxy','no_gain_normalization'},'sencache':{'official','latent_only','time_only'}}
 if row.get('variant','official') not in variants.get(row.get('method'),set()):raise ValueError('Unknown method-specific variant')
 if row.get('method') not in {'baseline','teacache','magcache','seacache','sencache'}: raise ValueError('Unknown method')
 for k in ['prompt','prompt_id','model_id','run_id']:
  if not isinstance(row.get(k),str) or not row[k]: raise ValueError(f'Missing {k}')
 if '/' in row['run_id'] or '\\' in row['run_id'] or row['run_id'] in {'.','..'}: raise ValueError('Unsafe run_id')
 for k in ['height','width','num_frames','num_inference_steps']:
  if not isinstance(row.get(k),int) or row[k]<=0: raise ValueError(f'Invalid {k}')
 if row['height']%16 or row['width']%16: raise ValueError('Wan dimensions must be divisible by16')
 if row['num_frames']%4!=1: raise ValueError('Wan num_frames must be4n+1')
 if row.get('dtype','bfloat16')!='bfloat16': raise ValueError('This backend is explicitly pinned to bfloat16 autocast')
 if row.get('sample_solver','unipc') not in {'unipc','dpm++'}: raise ValueError('Unknown solver')
 if not isinstance(row.get('seed'),int) or row['seed']<0: raise ValueError('Nonnegative fixed seed required')
 for k in ('guidance_scale','flow_shift'):
  if not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<=0: raise ValueError(f'Invalid {k}')
 if row['method']!='baseline':
  if not isinstance(row.get('threshold'),(int,float)) or not math.isfinite(row['threshold']) or row['threshold']<0: raise ValueError('nonnegative threshold required')
 if row.get('attention_backend','flash_attention_2')!='flash_attention_2':raise ValueError('Only pinned flash_attention_2 backend supported')
 if row.get('generation_backend','seacache_wan')!='seacache_wan':raise ValueError('This version uses the SeaCache Wan denoising loop; other backends are not implemented')
 if row.get('offload_model',False) or row.get('offload',False):raise ValueError('Benchmark requires all model components resident on GPU')
 if row.get('batch_size',1)!=1:raise ValueError('Only batch size1 supported')
 if '1.3B' not in row['model_id']:raise ValueError('This adapter supports Wan2.1-T2V-1.3B only')
 for k in ('threshold_start',):
  if k in row and (not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) or row[k]<0):raise ValueError(f'Invalid {k}')
 if 'max_skip' in row and (not isinstance(row['max_skip'],int) or row['max_skip']<0):raise ValueError('max_skip must be nonnegative integer')
 if 'retention_ratio' in row and (not isinstance(row['retention_ratio'],(int,float)) or not math.isfinite(row['retention_ratio']) or not 0<row['retention_ratio']<=1):raise ValueError('Invalid retention_ratio')
 return row
