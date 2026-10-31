"""Small CPU tensor parity/invariant tests of pinned ORIGINAL forward functions.
Not a full Wan/GPU or paper reproduction test. Need torch and fetched sources.
"""
import math, os, unittest, warnings
from pathlib import Path
try:
 import torch
 import numpy as np
except ImportError: torch=None
from cachebench.wan_adapter import extract_function,TEA_COEFFICIENTS

ROOT=Path(__file__).resolve().parents[3]
VENDOR=Path(os.environ.get('CACHEBENCH_VENDOR',Path(__file__).resolve().parents[1]/'vendor'))
LOCAL={k:ROOT/f'{k}_source' for k in ('teacache','magcache','seacache','sencache')}
def root_for(k):return VENDOR/k if (VENDOR/k).exists() else LOCAL[k]
AVAILABLE=torch is not None and all(root_for(k).exists() for k in LOCAL)

if torch is not None:
 class Head(torch.nn.Module):
  def forward(self,x,e):return x
 class Block(torch.nn.Module):
  def __init__(self):
   super().__init__();self.norm1=torch.nn.LayerNorm(8);self.modulation=torch.nn.Parameter(torch.zeros(1,6,8));self.calls=0
  def forward(self,x,e,context,**kw):
   self.calls+=1
   return x+.2*x+context.mean()+.01*e.mean()
 class Tiny(torch.nn.Module):
  def __init__(self):
   super().__init__();self.model_type='t2v';self.dim=self.freq_dim=8;self.text_len=4
   self.patch_embedding=torch.nn.Conv3d(1,8,1);self.freqs=torch.zeros(1)
   self.time_embedding=torch.nn.Linear(8,8);self.time_projection=torch.nn.Linear(8,48)
   self.text_embedding=torch.nn.Linear(4,8);self.blocks=torch.nn.ModuleList([Block(),Block()]);self.head=Head()
  def unpatchify(self,x,grid_sizes):return [x[0].clone()]
 def make_model(method,threshold=1e12):
  import types,importlib.util,logging
  torch.manual_seed(7);m=Tiny()
  filenames={'teacache':'TeaCache4Wan2.1/teacache_generate.py','magcache':'MagCache4Wan2.1/magcache_generate.py','seacache':'Wan2.1/seacache_generate.py','sencache':'Wan2.1/sencache.py'}
  ns={'torch':torch,'np':np,'amp':torch.cuda.amp,'logging':logging,
      'sinusoidal_embedding_1d':lambda dim,t:t[:,None].repeat(1,dim)/1000}
  if method=='seacache':
   spec=importlib.util.spec_from_file_location('sea_test_util',root_for(method)/'Wan2.1/util_seacache.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
   ns.update(apply_sea_with_scheduler=u.apply_sea_with_scheduler,rel_l1=u.rel_l1)
  f,_=extract_function(root_for(method)/filenames[method],method+'_forward',ns)
  m.forward=types.MethodType(f,m);m.cnt=0;m.num_steps=8
  if method in {'teacache','seacache'}:
   setattr(m,'enable_'+method,True);setattr(m,method+'_thresh',threshold)
   m.ret_steps=2;m.cutoff_steps=6
   for branch in ('even','odd'):
    setattr(m,'accumulated_rel_l1_distance_'+branch,0)
    setattr(m,'previous_e0_'+branch,None);setattr(m,'previous_residual_'+branch,None)
   if method=='teacache':m.use_ref_steps=False;m.coefficients=TEA_COEFFICIENTS[False]
   else:m.scheduler=types.SimpleNamespace(sigmas=torch.tensor([.9,.6,.3,.1]))
  elif method=='magcache':
   m.mag_ratios=np.ones(8);m.K=3;m.retention_ratio=.25;m.magcache_thresh=threshold
   m.residual_cache=[None,None];m.accumulated_ratio=[1.,1.];m.accumulated_err=[0.,0.];m.accumulated_steps=[0,0]
  else:
   m.cached_z=m.cached_t=m.cached_residual_cond=m.cached_residual_uncond=None
   m.cached_J_z_norm=m.cached_J_t_norm=None;m.accumulated_skips=0;m.skip_this_step=False
   m.current_skip_count=0;m.all_skip_counts=[];m.threshold_switch_step=2
   m.sencache_threshold_start=m.sencache_threshold_main=threshold;m.sencache_K=2
   m.ret_steps=0;m.cutoff_steps=8;m.J_z_norm=[1]*4;m.J_t_norm=[1]*4;m.sencache_timesteps_array=np.array([900,600,300,100])
  return m
 def call(m,x,t,cond):
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   return m([x],t=torch.tensor([t],dtype=torch.float32),context=[torch.full((4,4),.7 if cond else -.3)],seq_len=4)[0]

@unittest.skipUnless(AVAILABLE,'requires torch and pinned official sources fetched via cachebench.sources')
class OfficialForwardTensorTests(unittest.TestCase):
 def test_reuse_adds_old_residual_to_new_input_all_methods(self):
  for method in LOCAL:
   with self.subTest(method=method):
    m=make_model(method);x=torch.arange(4,dtype=torch.float32).reshape(1,1,2,2)
    old=call(m,x,900,True);call(m,x,900,False)
    initial=m.patch_embedding(x.unsqueeze(0)).flatten(2).transpose(1,2)[0]
    residual=old-initial
    newx=x+.03;newin=m.patch_embedding(newx.unsqueeze(0)).flatten(2).transpose(1,2)[0]
    before=m.blocks[0].calls
    out=call(m,newx,600,True)
    self.assertEqual(m.blocks[0].calls,before,'should reuse')
    torch.testing.assert_close(out,newin+residual)
    self.assertFalse(torch.equal(out,old),'current input must still change output')
 def test_cfg_residuals_are_separate(self):
  for method in LOCAL:
   m=make_model(method);x=torch.ones(1,1,2,2)
   oldc=call(m,x,900,True);oldu=call(m,x,900,False)
   yc=call(m,x,600,True);yu=call(m,x,600,False)
   self.assertFalse(torch.equal(yc,yu))
   torch.testing.assert_close(yc,oldc);torch.testing.assert_close(yu,oldu)
 def test_sen_conditional_decision_shared_and_k_respected(self):
  m=make_model('sencache');x=torch.ones(1,1,2,2)
  calls=[]
  for t in [900,600,300,100]:
   before=m.blocks[0].calls;call(m,x,t,True);call(m,x,t,False);calls.append(m.blocks[0].calls-before)
  self.assertEqual(calls,[2,0,0,2]);self.assertIsNone(m.cached_z)
 def test_zero_threshold_computes_every_call_nonpolynomial(self):
  for method in ('magcache','seacache','sencache'):
   m=make_model(method,threshold=0);x=torch.ones(1,1,2,2)
   for t in [900,600,300,100]:call(m,x,t,True);call(m,x,t,False)
   self.assertEqual(m.blocks[0].calls,8)
 def test_tea_zero_threshold_is_not_a_no_cache_switch(self):
  m=make_model('teacache',threshold=0);x=torch.ones(1,1,2,2)
  for t in [900,600,300,100]:call(m,x,t,True);call(m,x,t,False)
  self.assertEqual(m.blocks[0].calls,6) # polynomial may be negative; preserve official behavior
 def test_repeated_video_resets_cache_state(self):
  for method in LOCAL:
   m=make_model(method);x=torch.ones(1,1,2,2)
   passes=[]
   for _ in range(2):
    values=[]
    for t in [900,600,300,100]:values += [call(m,x,t,True),call(m,x,t,False)]
    passes.append(values)
   for first,second in zip(*passes):torch.testing.assert_close(first,second)
 def test_sea_filter_tensor_finite_shape_and_mean_gain(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('sea_util_test',root_for('seacache')/'Wan2.1/util_seacache.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
  x=torch.randn(1,3,4,6,2)
  y=u.apply_sea_from_ab(x,.5,.5,power_exp=3.,dims=(-2,-3,-4),norm_mode='mean')
  self.assertEqual(x.shape,y.shape);self.assertTrue(torch.isfinite(y).all())
  # Independently construct official separable frequency weighting.
  H=1
  for ax in (-2,-3,-4):
   f=torch.fft.fftfreq(x.shape[ax]);S=1/(f.abs()**3+1e-16)
   g=.5*S/(.25*S+.25+1e-16);sh=[1]*5;sh[ax]=len(g);H=H*g.reshape(sh)
  H=H/H.mean();expected=torch.fft.ifftn(torch.fft.fftn(x,dim=(-2,-3,-4))*H,dim=(-2,-3,-4)).real
  torch.testing.assert_close(y,expected)
 def test_official_81f_threshold_scale(self):
  self.assertAlmostEqual(math.sqrt(16*21*60*104),1447.97790038,places=7)
if __name__=='__main__':unittest.main()
