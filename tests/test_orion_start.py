"""Offline launcher checks: mock Slurm and model downloads; no GPU jobs or network."""
import argparse,contextlib,hashlib,importlib.util,io,json,sys,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/start_orion.py'
spec=importlib.util.spec_from_file_location('starter',SCRIPT);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
COLLECT_RESOURCES=s.collect_cluster_resources
class StarterTests(unittest.TestCase):
 def setUp(self):
  resource_patch=patch.object(s,'collect_cluster_resources');resource_patch.start();self.addCleanup(resource_patch.stop)
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.repo=self.root/'repo';self.repo.mkdir()
  self.files=['scripts/start_orion.py','requirements.txt','pyproject.toml','configs/correctness_smoke.json','src/wan_adapter.py']
  for f in self.files:
   p=self.repo/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# test\n')
  self.refresh();self.args=argparse.Namespace(work_root=self.root/'runtime',account='test_account',qos='highpriority',cpu_partition='cpup',gpu_partition='h200p',python_module='python/3.11.9',cuda_module='cuda/12.4.1',min_free_gb=0,submit=False,time=None)
 def tearDown(self):self.t.cleanup()
 def refresh(self):
  (self.repo/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256((self.repo/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in self.files))
 def test_checksum_mutation_rejected(self):
  (self.repo/'requirements.txt').write_text('bad')
  with self.assertRaisesRegex(ValueError,'checksum changed'):s.verified_release(self.repo)
 def test_escape_or_git_rejected(self):
  for name in ('../escape','.git/config'):
   with self.subTest(name=name):
    p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('a')
    (self.repo/'SHA256SUMS.txt').write_text(hashlib.sha256(b'a').hexdigest()+'  '+name+'\n')
    with self.assertRaisesRegex(ValueError,'Unsafe'):s.verified_release(self.repo)
 def test_preview_does_not_create_or_submit(self):
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()):s.submit(self.args)
  run.assert_not_called();self.assertFalse(self.args.work_root.exists())
 def test_repo_runtime_rejected(self):
  self.args.work_root=self.repo/'runtime'
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),self.assertRaisesRegex(ValueError,'outside'):s.submit(self.args)
 def test_submit_dependency_and_snapshot(self):
  self.args.submit=True
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.shutil,'which',return_value='/bin/sbatch'),patch.object(s,'run',side_effect=['101\n','102\n','103;orion\n']) as run,contextlib.redirect_stdout(io.StringIO()):s.submit(self.args)
  cmds=[x.args[0] for x in run.call_args_list]
  self.assertEqual(len(cmds),3);self.assertNotIn('--gres',cmds[0]);self.assertNotIn('--gres',cmds[1]);self.assertIn('gpu:1',cmds[2]);self.assertIn('afterok:101:102',cmds[2]);self.assertIn('--kill-on-invalid-dep=yes',cmds[2]);self.assertTrue(all('--time' not in command for command in cmds))
  d=next((self.args.work_root/'runs').iterdir());plan=json.loads((d/'plan.json').read_text());self.assertEqual(plan['jobs']['smoke'],'103');self.assertEqual(len(list((d/'code').rglob('*.*'))),6);self.assertTrue((d/'env.sh').read_text().startswith('#!/bin/bash -l'))
 def test_explicit_startup_time_is_passed_to_each_job(self):
  self.args.submit=True;self.args.time='1-00:00:00'
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.shutil,'which',return_value='/bin/sbatch'),patch.object(s,'run',side_effect=['101\n','102\n','103\n']) as run,contextlib.redirect_stdout(io.StringIO()):s.submit(self.args)
  for call in run.call_args_list:
   cmd=call.args[0];self.assertEqual(cmd[cmd.index('--time')+1],self.args.time)
 def test_array_defaults_have_no_additional_caps(self):
  self.array_command()
 def test_resource_query_failure_remains_visible(self):
  d=self.root/'resource_report';d.mkdir()
  result=types.SimpleNamespace(returncode=1,stdout='',stderr='permission denied')
  with patch.object(s.subprocess,'run',return_value=result):
   COLLECT_RESOURCES(d,{'cpu_partition':'cpup','gpu_partition':'h200p'})
  report=json.loads((d/'cluster_resources.json').read_text())
  self.assertEqual(len(report['queries']),9)
  self.assertTrue(all(x['returncode']==1 for x in report['queries']))
  self.assertTrue(all(x['command'][0] in {'sinfo','scontrol','sacctmgr','squeue'} for x in report['queries']))
  self.assertIn('do not mean unlimited',report['note'])
 def test_array_explicit_overrides_and_qos(self):
  self.array_command(['--concurrency','12','--time','1-00:00:00','--qos','highpriority'])
 def array_command(self,extras=None):
  path=SCRIPT.with_name('submit_slurm.py')
  spec=importlib.util.spec_from_file_location('array_launcher',path);array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  manifest=self.root/'manifest.jsonl';manifest.write_text('{}\n{}\n')
  argv=[str(path),'--manifest',str(manifest),'--output-dir',str(self.root/'out'),'--wan-repo',str(self.root/'vendor/wan'),'--checkpoint-dir',str(self.root/'model')]+(extras or [])
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run') as runner,contextlib.redirect_stdout(io.StringIO()) as out:array.main()
  runner.assert_not_called();result=json.loads(out.getvalue());cmd=result['command'];task_range=cmd[cmd.index('--array')+1]
  if extras:
   self.assertEqual(task_range,'0-1%12');self.assertEqual(cmd[cmd.index('--time')+1],'1-00:00:00');self.assertEqual(cmd[cmd.index('--qos')+1],'highpriority')
  else:
   self.assertEqual(task_range,'0-1');self.assertNotIn('--time',cmd);self.assertIsNone(result['max_parallel_gpus'])
 def test_partial_submit_records_jobs(self):
  self.args.submit=True
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.shutil,'which',return_value='/bin/sbatch'),patch.object(s,'run',side_effect=['101\n',RuntimeError('no quota')]),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'no quota'):s.submit(self.args)
  d=next((self.args.work_root/'runs').iterdir());self.assertEqual(json.loads((d/'plan.json').read_text())['jobs'],{'env':'101'})
 def test_copy_time_edit_rejected_before_submission(self):
  self.args.submit=True
  copy=s.shutil.copy2
  def changed_copy(source,destination):
   result=copy(source,destination)
   if Path(destination).name=='requirements.txt':Path(destination).write_text('changed during copy')
   return result
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.shutil,'which',return_value='/bin/sbatch'),patch.object(s.shutil,'copy2',side_effect=changed_copy),patch.object(s,'run') as run,self.assertRaisesRegex(ValueError,'checksum changed'):s.submit(self.args)
  run.assert_not_called()
 def test_status_bundle_has_no_weights_or_video(self):
  d=self.root/'run';d.mkdir();(d/'logs').mkdir();(d/'logs/env-1.log').write_text('log');(d/'video.mp4').write_text('video')
  (d/'plan.json').write_text(json.dumps({'jobs':{},'checkpoint':'model'}))
  with contextlib.redirect_stdout(io.StringIO()) as out:s.status(argparse.Namespace(run_dir=d,bundle=True))
  self.assertEqual(json.loads(out.getvalue())['status'],'NOT_READY')
  with zipfile.ZipFile(d/'startup_report.zip') as z:self.assertEqual(set(z.namelist()),{'plan.json','logs/env-1.log'})
 def model(self,bad=False):
  d=self.root/'run';d.mkdir();ck=self.root/'models/wan';ck.mkdir(parents=True)
  (d/'plan.json').write_text(json.dumps({'checkpoint':str(ck)}));items=[]
  for i,name in enumerate(s.MODEL_FILES):
   data=json.dumps({'model_type':'t2v','dim':1536,'num_layers':30,'in_dim':16,'out_dim':16}).encode() if name=='config.json' else b'fake bytes'
   p=ck/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
   lfs=types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if i<3 else None
   blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
   items.append(types.SimpleNamespace(rfilename=name,size=len(data),lfs=lfs,blob_id=blob))
  if bad:(ck/s.MODEL_FILES[0]).write_bytes(b'evil bytes')
  module=types.ModuleType('huggingface_hub');module.HfApi=lambda:types.SimpleNamespace(model_info=lambda *a,**kw:types.SimpleNamespace(sha=s.MODEL_REVISION,siblings=items));module.snapshot_download=lambda *a,**kw:None
  return d,module
 def test_model_integrity_checks_all_exact_files(self):
  d,m=self.model()
  with patch.dict(sys.modules,{'huggingface_hub':m}):s.download_model(d)
  report=json.loads((d/'model_integrity.json').read_text());self.assertEqual(len(report['files']),10);self.assertEqual(report['revision'],s.MODEL_REVISION)
 def test_same_size_model_corruption_rejected(self):
  d,m=self.model(True)
  with patch.dict(sys.modules,{'huggingface_hub':m}),self.assertRaisesRegex(ValueError,'hash mismatch'):s.download_model(d)
if __name__=='__main__':unittest.main(verbosity=2)
