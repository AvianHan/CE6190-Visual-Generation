"""Offline launcher checks: mock Slurm and model downloads; no GPU jobs or network."""
import argparse,contextlib,hashlib,importlib.util,io,json,os,subprocess,sys,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/start_orion.py'
spec=importlib.util.spec_from_file_location('starter',SCRIPT);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
COLLECT_RESOURCES=s.collect_cluster_resources
class StarterTests(unittest.TestCase):
 def setUp(self):
  resource_patch=patch.object(s,'collect_cluster_resources');resource_patch.start();self.addCleanup(resource_patch.stop)
  # macOS /var aliases /private/var; compare paths using one canonical root.
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name).resolve();self.repo=self.root/'repo';self.repo.mkdir()
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
  argv=[str(path),'--manifest',str(manifest),'--output-dir',str(self.root/'out'),'--wan-repo',str(self.root/'vendor/wan'),'--checkpoint-dir',str(self.root/'model'),'--max-array-size','1001']+(extras or [])
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
 def test_smoke_generates_both_rows_without_deleted_wrapper(self):
  d=self.root/'run';d.mkdir();(d/'code').mkdir()
  manifest=d/'smoke.jsonl';manifest.write_text(json.dumps({'method':'baseline','run_id':'ref'})+'\n'+json.dumps({'method':'seacache','run_id':'sea'})+'\n')
  with patch.object(s,'run') as run:s.generate_manifest_rows(d,manifest,d/'results',self.root/'model')
  self.assertEqual(run.call_count,2)
  for i,call in enumerate(run.call_args_list):
   cmd=list(map(str,call.args[0]))
   self.assertEqual(cmd[:3],[sys.executable,'-m','cachebench.generate'])
   self.assertEqual(cmd[cmd.index('--index')+1],str(i))
   self.assertEqual(cmd[cmd.index('--vendor-dir')+1],str(d/'vendor'))
   self.assertEqual(cmd[cmd.index('--checkpoint-dir')+1],str(self.root/'model'))
   self.assertFalse(any('run_manifest.py' in x for x in cmd))
   self.assertEqual(call.kwargs['cwd'],d/'code')
 def test_smoke_stops_if_first_generation_fails(self):
  d=self.root/'run';d.mkdir();manifest=d/'smoke.jsonl';manifest.write_text('{"method":"baseline"}\n{"method":"seacache"}\n')
  with patch.object(s,'run',side_effect=RuntimeError('generation failed')) as run,self.assertRaisesRegex(RuntimeError,'generation failed'):
   s.generate_manifest_rows(d,manifest,d/'results',self.root/'model')
  self.assertEqual(run.call_count,1)
 def recovery_fixture(self):
  self.files.append('configs/smoke_prompts.jsonl');(self.repo/'configs/smoke_prompts.jsonl').write_text('{"prompt":"A dog runs."}\n');self.refresh()
  d=self.root/'runtime/runs/existing';(d/'code').mkdir(parents=True);(d/'logs').mkdir()
  for name in self.files+['SHA256SUMS.txt']:
   path=d/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((self.repo/name).read_bytes())
  plan={'run_dir':str(d),'work_root':str(self.root/'runtime'),'account':'test_account','qos':'highpriority','cpu_partition':'cpup','gpu_partition':'h200p','python_module':'python/3.11.9','cuda_module':'cuda/12.4.1','checkpoint':str(self.root/'runtime/models/wan'),'jobs':{'env':'101','model':'102','smoke':'103'},'requested_time':None}
  (d/'plan.json').write_text(json.dumps(plan))
  args=argparse.Namespace(run_dir=d,submit=False,queue_development=False,time=None)
  return d,args
 def test_resume_preview_reuses_run_without_creating_recovery(self):
  d,args=self.recovery_fixture();before={str(p.relative_to(d)):p.read_bytes() for p in d.rglob('*') if p.is_file()}
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(stdout='',returncode=0)),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()):s.resume_smoke(args)
  run.assert_not_called();after={str(p.relative_to(d)):p.read_bytes() for p in d.rglob('*') if p.is_file()};self.assertEqual(before,after)
 def test_resume_refuses_active_old_smoke(self):
  d,args=self.recovery_fixture();args.submit=True
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(stdout='103\n',returncode=0)),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,RuntimeError)):s.resume_smoke(args)
  run.assert_not_called();self.assertEqual(json.loads((d/'plan.json').read_text())['jobs']['smoke'],'103')
 def test_resume_rejects_changed_runtime_source(self):
  d,args=self.recovery_fixture();args.submit=True;(self.repo/'src/wan_adapter.py').write_text('# incompatible source\n');self.refresh()
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(stdout='',returncode=0)),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,RuntimeError)):s.resume_smoke(args)
  run.assert_not_called();self.assertEqual(json.loads((d/'plan.json').read_text())['jobs']['smoke'],'103')
 def test_resume_does_not_overwrite_passed_smoke(self):
  d,args=self.recovery_fixture();args.submit=True;passed=json.dumps({'status':'passed','job_id':'103'});(d/'smoke.json').write_text(passed)
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(stdout='',returncode=0)),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,RuntimeError)):s.resume_smoke(args)
  run.assert_not_called();self.assertEqual((d/'smoke.json').read_text(),passed)
 def test_resume_submits_only_smoke_with_existing_dependencies(self):
  d,args=self.recovery_fixture();args.submit=True
  partial=d/'smoke_results/partial';partial.mkdir(parents=True);(partial/'trace.json').write_text('previous diagnostic');(d/'smoke.json').write_text('{"status":"failed"}')
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.shutil,'which',return_value='/bin/sbatch'),patch.object(s.subprocess,'run',return_value=types.SimpleNamespace(stdout='',returncode=0)),patch.object(s,'run',return_value='104\n') as run,contextlib.redirect_stdout(io.StringIO()):s.resume_smoke(args)
  self.assertEqual(run.call_count,1);cmd=list(map(str,run.call_args.args[0]));self.assertEqual(cmd[0],'sbatch');self.assertIn('afterok:101:102',cmd);self.assertNotIn('--time',cmd)
  plan=json.loads((d/'plan.json').read_text());self.assertEqual(plan['jobs'],{'env':'101','model':'102','smoke':'104'})
  self.assertEqual(plan['smoke_history'][-1]['previous_smoke_job'],'103');self.assertEqual(plan['smoke_history'][-1]['job_id'],'104')
  self.assertTrue(any(p.name=='start_orion.py' for p in (d/'recovery').rglob('*')))
  preserved=list((d/'recovery').rglob('previous_attempt/smoke_results/partial/trace.json'));self.assertEqual(len(preserved),1);self.assertEqual(preserved[0].read_text(),'previous diagnostic');self.assertFalse((d/'smoke_results').exists())
  wrapper=Path(cmd[-1]).read_text();self.assertIn(str(d),wrapper);self.assertIn('_phase',wrapper);self.assertIn('smoke',wrapper)
  self.assertFalse((d/'venv').exists());self.assertFalse((d/'download_venv').exists())
 def test_resume_refuses_failed_environment_without_resubmitting_anything(self):
  d,args=self.recovery_fixture();args.submit=True;(d/'env.json').write_text('{"status":"failed"}')
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'Prerequisite failed'):
   s.resume_smoke(args)
  run.assert_not_called();self.assertFalse((d/'recovery').exists())
 def array_preview(self,count,extras):
  path=SCRIPT.with_name('submit_slurm.py');spec=importlib.util.spec_from_file_location('array_launcher',path);array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  manifest=self.root/"manifest 'with spaces.jsonl";manifest.write_text('{}\n'*count)
  argv=[str(path),'--manifest',str(manifest),'--output-dir',str(self.root/'results with spaces'),'--wan-repo',str(self.root/'vendor with spaces/wan'),'--checkpoint-dir',str(self.root/'model with spaces')]+extras
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run') as runner,contextlib.redirect_stdout(io.StringIO()) as out:array.main()
  runner.assert_not_called();self.assertFalse((self.root/'results with spaces').exists())
  return json.loads(out.getvalue())
 def test_array_chunking_covers_all_1536_runs(self):
  result=self.array_preview(1536,['--max-array-size','1001','--dependency','afterok:104'])
  self.assertEqual([(x['offset'],x['count']) for x in result['arrays']],[(0,1001),(1001,535)])
  self.assertEqual([x['array'] for x in result['arrays']],['0-1000','0-534'])
  for command in result['commands']:
   self.assertIn('--wrap',command);self.assertFalse(any('slurm_job.sh' in part for part in command));self.assertIn('afterok:104',command)
   self.assertNotIn('--time',command);self.assertNotIn('%',command[command.index('--array')+1])
 def test_array_max_task_count_respected_separately_from_index_limit(self):
  result=self.array_preview(25,['--max-array-size','1001','--max-array-tasks','10'])
  self.assertEqual([(x['offset'],x['count']) for x in result['arrays']],[(0,10),(10,10),(20,5)])
 def test_array_wrapper_quotes_paths_and_offsets_task_index(self):
  fake=self.root/"fake python 'with spaces";fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["CE6190_TEST_ARGV"],"w") as f: json.dump(sys.argv[1:],f)\n');fake.chmod(0o700)
  result=self.array_preview(25,['--max-array-size','11','--python-executable',str(fake)])
  command=result['commands'][1];wrapper=command[command.index('--wrap')+1];received=self.root/'received.json'
  subprocess.run(['/bin/bash','-c',wrapper],check=True,env={**os.environ,'SLURM_ARRAY_TASK_ID':'2','CE6190_TEST_ARGV':str(received)})
  argv=json.loads(received.read_text());self.assertEqual(argv[:2],['-m','cachebench.generate']);self.assertEqual(argv[argv.index('--index')+1],'13')
  self.assertEqual(argv[argv.index('--manifest')+1],result['manifest_snapshot']);self.assertEqual(argv[argv.index('--vendor-dir')+1],str(self.root/'vendor with spaces'))
 def test_array_reads_both_site_limits(self):
  spec=importlib.util.spec_from_file_location('array_launcher',SCRIPT.with_name('submit_slurm.py'));array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  args=argparse.Namespace(max_array_size=None,max_array_tasks=None)
  result=types.SimpleNamespace(stdout='MaxArraySize = 1001\nSchedulerParameters = bf_continue,max_array_tasks=999\n',returncode=0)
  with patch.object(array.subprocess,'run',return_value=result):limits=array.array_limits(args)
  self.assertEqual(limits['MaxArraySize'],1001);self.assertEqual(limits['chunk_size'],999)
  with patch.object(array.subprocess,'run',side_effect=OSError('not available')),self.assertRaisesRegex(RuntimeError,'Cannot query'):array.array_limits(args)
 def test_array_partial_submission_preserves_receipt_and_prevents_duplicates(self):
  path=SCRIPT.with_name('submit_slurm.py');spec=importlib.util.spec_from_file_location('array_launcher',path);array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  manifest=self.root/'manifest.jsonl';content='{}\n'*25;manifest.write_text(content);output=self.root/'out'
  argv=[str(path),'--manifest',str(manifest),'--output-dir',str(output),'--wan-repo',str(self.root/'vendor/wan'),'--checkpoint-dir',str(self.root/'model'),'--max-array-size','10','--submit']
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run',side_effect=[types.SimpleNamespace(stdout='201;orion\n'),RuntimeError('quota denied')]) as runner,contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'quota denied'):array.main()
  self.assertEqual(runner.call_count,2);receipt=json.loads((output/'slurm_submission.json').read_text());self.assertEqual(receipt['status'],'partial_submission');self.assertEqual(receipt['submitted_jobs'],['201']);self.assertEqual((output/'slurm_manifest.jsonl').read_text(),content)
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run') as runner,self.assertRaisesRegex(RuntimeError,'uncertain'):array.main()
  runner.assert_not_called()
 def test_array_retries_only_known_rejected_chunks(self):
  path=SCRIPT.with_name('submit_slurm.py');spec=importlib.util.spec_from_file_location('array_launcher',path);array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  manifest=self.root/'manifest.jsonl';manifest.write_text('{}\n'*25);output=self.root/'out'
  argv=[str(path),'--manifest',str(manifest),'--output-dir',str(output),'--wan-repo',str(self.root/'vendor/wan'),'--checkpoint-dir',str(self.root/'model'),'--max-array-size','10','--submit']
  failure=subprocess.CalledProcessError(1,['sbatch'],stderr='Job submit limit')
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run',side_effect=[types.SimpleNamespace(stdout='201\n'),failure]),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(subprocess.CalledProcessError):array.main()
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run',side_effect=[types.SimpleNamespace(stdout='202\n'),types.SimpleNamespace(stdout='203\n')]) as runner,contextlib.redirect_stdout(io.StringIO()):array.main()
  self.assertEqual(runner.call_count,2);receipt=json.loads((output/'slurm_submission.json').read_text());self.assertEqual(receipt['submitted_jobs'],['201','202','203']);self.assertEqual(receipt['status'],'submitted')
  self.assertEqual([x['job_id'] for x in receipt['arrays']],['201','202','203'])
  with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run') as runner,contextlib.redirect_stdout(io.StringIO()) as out:array.main()
  runner.assert_not_called();self.assertEqual(json.loads(out.getvalue())['status'],'already_submitted')
 def test_development_queues_two_separate_complete_grids_after_smoke(self):
  d,args=self.recovery_fixture();args.submit=True;args.partitions=['h200p','l40sp'];repo=SCRIPT.parents[1]
  with patch.object(s,'compatible_runtime'),patch.object(s.shutil,'disk_usage',return_value=types.SimpleNamespace(free=2**50)),patch.object(s,'run') as run,contextlib.redirect_stdout(io.StringIO()):s.queue_development(args,repo=repo)
  self.assertEqual(run.call_count,2)
  for partition,call in zip(args.partitions,run.call_args_list):
   rows=[json.loads(line) for line in (d/('development_'+partition)/'manifest.jsonl').read_text().splitlines()]
   self.assertEqual(len(rows),1536);self.assertEqual({x['method'] for x in rows},{'baseline','teacache','seacache'});self.assertEqual(len({x['prompt_id'] for x in rows}),64);self.assertEqual({x['seed'] for x in rows},{0,1,2})
   references={x['run_id'] for x in rows if x['method']=='baseline'};self.assertTrue(all(x['reference_run_id'] in references for x in rows))
   cmd=list(map(str,call.args[0]));self.assertEqual(cmd[cmd.index('--partition')+1],partition);self.assertEqual(cmd[cmd.index('--dependency')+1],'afterok:103');self.assertEqual(cmd[cmd.index('--require-smoke')+1],str(d/'smoke.json'));self.assertEqual(cmd[cmd.index('--python-executable')+1],str(d/'venv/bin/python'));self.assertNotIn('--concurrency',cmd)
if __name__=='__main__':unittest.main(verbosity=2)
