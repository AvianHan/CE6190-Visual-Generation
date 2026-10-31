import math, unittest
from cachebench.policies import AccumulatedDistance,MagnitudeRule,sensitivity_score,polynomial,validate_row,config_signature
from cachebench.calibrate import aggregate_sen

class RuleTests(unittest.TestCase):
 def test_accumulate_and_reset(self):
  rule=AccumulatedDistance(.10)
  self.assertEqual([rule.decide(x) for x in [.03,.04,.05]],[False,False,True])
  self.assertEqual(rule.accumulated,0)
  self.assertFalse(rule.decide(.01))
 def test_threshold_equality_refreshes(self):self.assertTrue(AccumulatedDistance(.1).decide(.1))
 def test_mag_product_and_sum(self):
  r=MagnitudeRule(.2,4)
  self.assertFalse(r.decide(.9))
  self.assertTrue(r.decide(.9)) # .1 + |1-.81| = .29
  self.assertEqual((r.ratio,r.error,r.steps),(1.,0.,0))
 def test_mag_max_skip(self):
  r=MagnitudeRule(1,2)
  self.assertEqual([r.decide(1) for _ in range(3)],[False,False,True])
 def test_sensitivity_last_reference_units(self):
  self.assertAlmostEqual(sensitivity_score(2,.1,.03,1),.16)
 def test_polynomial_matches_known(self):self.assertEqual(polynomial([2,3,4],2),18)
 def test_bad_values_rejected(self):
  for bad in [math.nan,math.inf,-1]:
   with self.assertRaises(ValueError):sensitivity_score(bad,1,1,1)
 def test_sen_raw_conversion(self):
  d=aggregate_sen([[[1,2],[3,4]],[[3,4],[5,6]]],[1000,500])
  self.assertEqual(d['J_x_norm'],[2,4]);self.assertEqual(d['J_t_norm'],[3,5])
 def test_sen_raw_failed_zero_rejected(self):
  with self.assertRaises(ValueError):aggregate_sen([[[0,0]]],[1000])
 def test_dimension_contract(self):
  row=dict(method='baseline',model_id='Wan-AI/Wan2.1-T2V-1.3B',run_id='r',prompt_id='p',prompt='dog',height=480,width=832,num_frames=81,num_inference_steps=50,guidance_scale=6,flow_shift=5,seed=0)
  validate_row(row)
  with self.assertRaises(ValueError):validate_row(dict(row,num_frames=80))
  with self.assertRaises(ValueError):validate_row(dict(row,method='sencache',threshold=float('nan')))
  with self.assertRaises(ValueError):validate_row(dict(row,run_id='../oops'))
if __name__=='__main__':unittest.main()

class VariantTests(unittest.TestCase):
 def test_method_variant_validation(self):
  row=dict(method='seacache',variant='raw_proxy',threshold=.2,model_id='Wan2.1-T2V-1.3B',run_id='r',prompt_id='p',prompt='dog',height=480,width=832,num_frames=81,num_inference_steps=50,guidance_scale=6,flow_shift=5,seed=0)
  validate_row(row)
  with self.assertRaises(ValueError):validate_row(dict(row,method='teacache'))
 def test_sensitivity_ablations_do_not_mutate_profile(self):
  import numpy as np
  from cachebench.wan_adapter import sensitivity_ablation
  x=np.array([1.,2.]);t=np.array([3.,4.])
  xx,tt=sensitivity_ablation(x,t,'latent_only')
  np.testing.assert_equal(xx,x);np.testing.assert_equal(tt,[0,0]);np.testing.assert_equal(t,[3,4])
  xx,tt=sensitivity_ablation(x,t,'time_only')
  np.testing.assert_equal(xx,[0,0]);np.testing.assert_equal(tt,t)
 def test_sea_hook_semantics(self):
  from cachebench.wan_adapter import sea_filter_hook
  seen={}
  def original(x,**kwargs):seen.update(kwargs);return x+10
  self.assertEqual(sea_filter_hook(original,'raw_proxy')(5,norm_mode='mean'),5)
  self.assertEqual(sea_filter_hook(original,'no_gain_normalization')(5,norm_mode='mean'),15)
  self.assertEqual(seen['norm_mode'],'none')
