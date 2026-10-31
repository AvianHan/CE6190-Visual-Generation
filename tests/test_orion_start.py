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
 def test_venv_pip_ignores_module_pythonpath_pollution(self):
  venv=self.root/'venv with spaces'
  subprocess.run([sys.executable,'-I','-m','venv',str(venv)],check=True,capture_output=True,text=True)
  poison=self.root/'module packages';(poison/'pip').mkdir(parents=True)
  (poison/'pip/__init__.py').write_text('raise RuntimeError("external module pip imported")\n')
  with patch.dict(os.environ,{'PYTHONPATH':str(poison),'PYTHONHOME':str(self.root/'invalid home')}),contextlib.redirect_stdout(io.StringIO()):
   report=s.check_venv_pip(venv/'bin/python')
  self.assertTrue(Path(report['pip_file']).is_relative_to(venv));self.assertEqual(Path(report['prefix']),venv)
 def test_runtime_keeps_proxy_and_certificate_but_removes_package_redirects(self):
  d=self.root/'runtime/runs/new';d.mkdir(parents=True)
  dirty={'PYTHONPATH':'spack','PYTHONHOME':'wrong','PIP_INDEX_URL':'https://invalid.example','PIP_TARGET':'outside','PIP_EXTRA_INDEX_URL':'https://invalid.example','HTTPS_PROXY':'https://proxy.example','PIP_CERT':'/test/cert.pem'}
  with patch.dict(os.environ,dirty):
   s.runtime_environment(d)
   for key in ('PYTHONPATH','PYTHONHOME','PIP_INDEX_URL','PIP_TARGET','PIP_EXTRA_INDEX_URL'):self.assertNotIn(key,os.environ)
   self.assertEqual(os.environ['HTTPS_PROXY'],dirty['HTTPS_PROXY']);self.assertEqual(os.environ['PIP_CERT'],dirty['PIP_CERT']);self.assertEqual(os.environ['PIP_CONFIG_FILE'],os.devnull)
 def test_module_wrapper_clears_paths_after_loading_modules(self):
  path=SCRIPT.with_name('submit_slurm.py');spec=importlib.util.spec_from_file_location('array_clean',path);array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  fake=self.root/'capturepython';received=self.root/'module_env.json'
  fake.write_text('#!/usr/bin/env python3\nimport os,json\nwith open(os.environ["CE6190_TEST_ENV"],"w") as f: json.dump({k:os.environ.get(k) for k in ("PYTHONPATH","PYTHONHOME")},f)\n');fake.chmod(0o700)
  args=argparse.Namespace(wan_repo=str(self.root/'vendor/wan'),python_executable=str(fake),python_module='python/test',cuda_module='cuda/test',chdir=str(self.root),cpus=8,runtime_dir=None,require_smoke=None,output_dir=str(self.root/'out'),checkpoint_dir=str(self.root/'weights'))
  body='module() { export PYTHONPATH=/spack/site-packages PYTHONHOME=/invalid; }\n'+array.batch_body(args,self.root/'manifest.jsonl',0)
  subprocess.run(['/bin/bash','--noprofile','--norc','-c',body],check=True,env={**os.environ,'SLURM_ARRAY_TASK_ID':'0','CE6190_TEST_ENV':str(received)})
  self.assertEqual(json.loads(received.read_text()),{'PYTHONPATH':None,'PYTHONHOME':None})
 def reusable_model_fixture(self):
  d,m=self.model()
  with patch.dict(sys.modules,{'huggingface_hub':m}):s.download_model(d)
  old=self.root/'models/wan';target=self.args.work_root/'models'/('Wan2.1-T2V-1.3B-'+s.MODEL_REVISION[:12]);target.parent.mkdir(parents=True);old.rename(target)
  (d/'plan.json').write_text(json.dumps({'checkpoint':str(target),'model_id':s.MODEL_ID,'model_revision':s.MODEL_REVISION}))
  report=json.loads((d/'model_integrity.json').read_text());report['checkpoint']=str(target);(d/'model_integrity.json').write_text(json.dumps(report))
  (d/'model.json').write_text('{"status":"passed","job_id":"184994"}')
  return d,target
 def test_reuse_model_creates_only_env_and_smoke_and_keeps_old_run(self):
  old,model=self.reusable_model_fixture();before={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
  self.args.reuse_model_from=old;self.args.submit=True;self.args.time='3-00:00:00'
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s.shutil,'which',return_value='/bin/sbatch'),patch.object(s,'run',side_effect=['501\n','502\n']) as runner,contextlib.redirect_stdout(io.StringIO()):s.submit(self.args)
  self.assertEqual(runner.call_count,2);cmd=list(map(str,runner.call_args_list[-1].args[0]));self.assertIn('afterok:501',cmd);self.assertFalse(any('184994' in x for x in cmd))
  new=next((self.args.work_root/'runs').iterdir());plan=json.loads((new/'plan.json').read_text());self.assertEqual(plan['jobs'],{'env':'501','smoke':'502'});self.assertTrue(plan['model_reuse']['hashes_verified']);self.assertEqual(json.loads((new/'model.json').read_text())['status'],'passed')
  self.assertFalse((new/'model.sh').exists());self.assertEqual(before,{str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()})
 def test_reuse_model_rejects_same_size_corruption_before_any_job(self):
  old,model=self.reusable_model_fixture();(model/s.MODEL_FILES[0]).write_bytes(b'evil bytes');self.args.reuse_model_from=old;self.args.submit=True
  with patch.object(s,'__file__',str(self.repo/'scripts/start_orion.py')),patch.object(s,'run') as runner,contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'hash mismatch'):s.submit(self.args)
  runner.assert_not_called();self.assertFalse((self.args.work_root/'runs').exists())
 def test_reuse_model_rejects_failed_preparation(self):
  old,model=self.reusable_model_fixture();(old/'model.json').write_text('{"status":"failed"}')
  with self.assertRaisesRegex(ValueError,'must have passed'):s.reusable_model(old,model,verify_hashes=True)
 def test_named_arrays_share_outputs_without_reusing_wrong_receipt(self):
  path=SCRIPT.with_name('submit_slurm.py');spec=importlib.util.spec_from_file_location('array_names',path);array=importlib.util.module_from_spec(spec);spec.loader.exec_module(array)
  output=self.root/'shared';args=['--output-dir',str(output),'--wan-repo',str(self.root/'vendor/wan'),'--checkpoint-dir',str(self.root/'model'),'--max-array-size','1001','--submit']
  commands=[]
  for name,job in [('magcache','601'),('sencache','602')]:
   manifest=self.root/(name+'.jsonl');manifest.write_text(json.dumps({'run_id':name})+'\n');argv=[str(path),'--manifest',str(manifest),'--submission-name',name]+args;commands.append(argv)
   with patch.object(sys,'argv',argv),patch.object(array.subprocess,'run',return_value=types.SimpleNamespace(stdout=job+'\n')),contextlib.redirect_stdout(io.StringIO()):array.main()
  receipts=list(output.glob('*submission.json'));self.assertEqual(len(receipts),2);self.assertEqual({json.loads(p.read_text())['submitted_jobs'][0] for p in receipts},{'601','602'})
  with patch.object(sys,'argv',commands[0]),patch.object(array.subprocess,'run') as runner,contextlib.redirect_stdout(io.StringIO()):array.main()
  runner.assert_not_called()
 def test_inherited_time_is_recorded_in_actual_array_command(self):
  with patch.dict(os.environ,{'SBATCH_TIMELIMIT':'3-00:00:00'}):result=self.array_preview(1,['--max-array-size','1001'])
  cmd=result['command'];self.assertEqual(cmd[cmd.index('--time')+1],'3-00:00:00');self.assertNotEqual(result['time_policy'],'Slurm policy default')

 def calibration_module(self):
  name='_ce6190_test_calibration';package=types.ModuleType(name);package.__path__=[str(SCRIPT.parents[1]/'src')];sys.modules[name]=package
  spec=importlib.util.spec_from_file_location(name+'.calibrate',SCRIPT.parents[1]/'src/calibrate.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module
 def calibration_fixture(self,method):
  c=self.calibration_module();row=json.loads((SCRIPT.parents[1]/'configs/validation_grid.json').read_text())['generation'];row['num_inference_steps']=2
  path=self.root/'samples.jsonl';values=[{'prompt_id':'cal0','prompt':'A dog walks beside a wooden fence.','split':'calibration','seed':0,'video_path':str(self.root/'dog.webm')},{'prompt_id':'cal1','prompt':'Water flows through a stone channel.','split':'calibration','seed':1,'video_path':str(self.root/'water.webm')}]
  path.write_text(''.join(json.dumps(x)+'\n' for x in values));samples=c.read_samples(path,method);paths=[]
  for i in range(2):
   data=c._profile_header(method,row,samples,[i],path,[])
   data.update(checkpoint_fingerprint='a'*64,source_revisions={'wan':'test-pin'},aggregation='test mean',calibration_wall_seconds=10+i)
   if method=='magcache':
    raw=[1.0,1.0,1.0+i,2.0+i];data.update(raw_ratios=[raw],mag_ratios=raw)
   else:
    raw=[[1.0+i,2.0+2*i],[3.0+3*i,4.0+4*i]];data.update(raw_data=[raw],J_x_norm=[p[0] for p in raw],J_t_norm=[p[1] for p in raw],timesteps=[1000.0,500.0],calibration_code_sha256='b'*64,sample_video_sha256={samples[i]['prompt_id']:str(i+1)*64})
   output=self.root/(method+str(i)+'.json');output.write_text(json.dumps(data));paths.append(output)
  return c,row,path,samples,paths
 def test_calibration_merge_restores_sample_order_without_gpu_libraries(self):
  for method in ('magcache','sencache'):
   with self.subTest(method=method):
    c,row,path,samples,paths=self.calibration_fixture(method)
    with patch.dict(sys.modules,{'numpy':None,'torch':None}):result=c.merge_profiles(paths[::-1],method,row,samples,path,[])
    self.assertEqual(result['sample_ids'],['cal0','cal1']);self.assertIsNone(result['calibration_wall_seconds']);self.assertEqual(result['calibration_time_accounting']['sum_shard_wall_seconds'],21)
    if method=='magcache':self.assertEqual(result['mag_ratios'],[1.0,1.0,1.5,2.5])
    else:self.assertEqual(result['J_x_norm'],[1.5,4.5]);self.assertEqual(result['J_t_norm'],[3.0,6.0])
 def test_calibration_merge_rejects_missing_duplicate_and_mixed_weights(self):
  c,row,path,samples,paths=self.calibration_fixture('magcache')
  for chosen in (paths[:1],[paths[0],paths[0]]):
   with self.subTest(chosen=chosen),self.assertRaisesRegex(ValueError,'Missing|Duplicate'):c.merge_profiles(chosen,'magcache',row,samples,path,[])
  changed=json.loads(paths[1].read_text());changed['checkpoint_fingerprint']='c'*64;paths[1].write_text(json.dumps(changed))
  with self.assertRaisesRegex(ValueError,'provenance differs'):c.merge_profiles(paths,'magcache',row,samples,path,[])
 def test_calibration_merge_checks_raw_values_and_video_uniqueness(self):
  c,row,path,samples,paths=self.calibration_fixture('sencache');changed=json.loads(paths[1].read_text());changed['J_x_norm'][0]=100;paths[1].write_text(json.dumps(changed))
  with self.assertRaisesRegex(ValueError,'retained raw'):c.merge_profiles(paths,'sencache',row,samples,path,[])
  c,row,path,samples,paths=self.calibration_fixture('sencache');changed=json.loads(paths[1].read_text());changed['sample_video_sha256']['cal1']='1'*64;paths[1].write_text(json.dumps(changed))
  with self.assertRaisesRegex(ValueError,'identical video'):c.merge_profiles(paths,'sencache',row,samples,path,[])
 def test_calibration_prompt_leakage_is_checked_for_the_entire_set(self):
  c,row,path,samples,paths=self.calibration_fixture('magcache');held=self.root/'held.jsonl';held.write_text(json.dumps({'prompt':'  WATER flows through a stone channel. '})+'\n')
  with self.assertRaisesRegex(ValueError,'overlap|leak|Overlap'):c.validate_exclusions(samples,[held])
 def test_profile_output_does_not_overwrite_previous_calibration(self):
  c=self.calibration_module();path=self.root/'profile.json';c.write_profile_atomic(path,{'test':1})
  with self.assertRaises(FileExistsError):c.write_profile_atomic(path,{'test':2})
  self.assertEqual(json.loads(path.read_text()),{'test':1})


class CalibratedQueueTests(unittest.TestCase):
    # Reuse filesystem setup without inheriting/running the original 42 tests twice.
    setUp = StarterTests.setUp
    tearDown = StarterTests.tearDown
    refresh = StarterTests.refresh

    def queue_fixture(self):
        sources = ['scripts/start_orion.py', 'scripts/submit_slurm.py', 'src/manifest.py',
                   'configs/validation_grid.json', 'configs/correctness_smoke.json',
                   'configs/development_prompts.jsonl', 'configs/smoke_prompts.jsonl',
                   'benchmarks/vbench/VBench_full_info.json']
        for name in sources:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((SCRIPT.parents[1] / name).read_bytes())
            if name not in self.files:
                self.files.append(name)
        self.refresh()
        directory = self.root / 'runtime/runs/five-methods'
        (directory / 'logs').mkdir(parents=True)
        for name in self.files + ['SHA256SUMS.txt']:
            target = directory / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.repo / name).read_bytes())
        plan = {'run_dir': str(directory), 'work_root': str(self.root / 'runtime'),
                'account': 'test_account', 'qos': 'highpriority', 'cpu_partition': 'l40sp',
                'gpu_partition': 'l40sp', 'python_module': 'python/3.11.9',
                'cuda_module': 'cuda/12.4.1', 'checkpoint': str(self.root / 'runtime/models/wan'),
                'jobs': {'env': '101', 'model': '102', 'smoke': '103'},
                'requested_time': '3-00:00:00'}
        (directory / 'plan.json').write_text(json.dumps(plan))
        for name in ('env', 'model'):
            (directory / (name + '.json')).write_text(json.dumps({'status': 'passed'}))
        self.samples = self.root / 'calibration.jsonl'
        self.sample_rows = [{'prompt_id': f'calibration-{index}',
                             'prompt': f'A dog explores calibration location number {index}.',
                             'seed': index, 'split': 'calibration',
                             'video_url': f'https://example.invalid/clip{index}.webm'}
                            for index in range(8)]
        self.write_samples(self.sample_rows)
        args = argparse.Namespace(run_dir=directory, partitions=['l40sp'], submit=True,
                                  with_calibration=True, calibration_samples=self.samples)
        return directory, plan, args

    def write_samples(self, rows):
        self.samples.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def fake_slurm(self):
        """Execute real nested array planning; replace only process/cluster boundaries."""
        spec = importlib.util.spec_from_file_location('queue_array_test', self.repo / 'scripts/submit_slurm.py')
        array = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(array)
        self.accepted = []
        self.nested_commands = []

        def process(command, *positional, **keywords):
            command = list(map(str, command))
            if command[:3] == ['scontrol', 'show', 'config']:
                return types.SimpleNamespace(stdout='MaxArraySize = 1001\n', returncode=0)
            if command[0] == 'squeue':
                ids = command[command.index('-j') + 1].split(',')
                return types.SimpleNamespace(stdout=''.join(job + '|PENDING|Dependency\n' for job in ids), returncode=0)
            if command[0] == 'sbatch':
                self.accepted.append(command)
                return types.SimpleNamespace(stdout=str(700 + len(self.accepted)) + '\n', returncode=0)
            if len(command) > 1 and command[1] == str(self.repo / 'scripts/submit_slurm.py'):
                self.nested_commands.append(command)
                with patch.object(sys, 'argv', command[1:]):
                    array.main()
                return types.SimpleNamespace(stdout='', returncode=0)
            self.fail(f'Unexpected external process: {command}')
        return process

    def queue(self, args):
        with patch.object(s.subprocess, 'run', side_effect=self.fake_slurm()), \
             patch.object(s.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=2**50)), \
             contextlib.redirect_stdout(io.StringIO()):
            s.queue_development(args, repo=self.repo)

    def test_full_grid_preserves_pairs_dependencies_and_submission_receipts(self):
        directory, plan, args = self.queue_fixture()
        process = self.fake_slurm()
        with patch.object(s.subprocess, 'run', side_effect=process), \
             patch.object(s.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=2**50)), \
             contextlib.redirect_stdout(io.StringIO()):
            s.queue_development(args, repo=self.repo)
            first_commands = list(self.accepted)
            s.queue_development(args, repo=self.repo)
        self.assertEqual(self.accepted, first_commands, 'A repeated request must not submit duplicate jobs')
        self.assertEqual(len(self.accepted), 8)  # 2 calibration arrays + 2 finalizers + 4 experiment chunks.
        development = directory / 'development_l40sp'
        groups = {'base': 'manifest.jsonl', 'magcache': 'magcache_manifest.jsonl', 'sencache': 'sencache_manifest.jsonl'}
        rows = {name: [json.loads(line) for line in (development / filename).read_text().splitlines()]
                for name, filename in groups.items()}
        self.assertEqual({name: len(values) for name, values in rows.items()},
                         {'base': 1536, 'magcache': 576, 'sencache': 768})
        all_rows = sum(rows.values(), [])
        self.assertEqual(len({row['run_id'] for row in all_rows}), 2880)
        self.assertEqual(len({row['prompt_id'] for row in all_rows}), 64)
        self.assertEqual({row['seed'] for row in all_rows}, {0, 1, 2})
        references = {row['run_id']: row for row in rows['base'] if row['method'] == 'baseline'}
        self.assertEqual(len(references), 192)
        shared_generation = json.loads((self.repo / 'configs/validation_grid.json').read_text())['generation']
        for row in all_rows:
            self.assertIn(row['reference_run_id'], references)
            baseline = references[row['reference_run_id']]
            self.assertEqual((row['prompt'], row['seed'], row['pair_id']),
                             (baseline['prompt'], baseline['seed'], baseline['pair_id']))
            for name, value in shared_generation.items():
                self.assertEqual(row[name], value)
            if row['method'] in ('magcache', 'sencache'):
                self.assertEqual(Path(row['calibration_path']), directory / 'calibration' / row['method'] / 'profile.json')
                self.assertTrue(Path(row['calibration_path']).is_absolute())
        for method in ('magcache', 'sencache'):
            root = directory / 'calibration' / method
            shard = json.loads((root / 'array-0.json').read_text())
            finalizer = json.loads((root / 'finalize-job.json').read_text())
            self.assertEqual(shard['command'][shard['command'].index('--array') + 1], '0-7')
            self.assertEqual(shard['command'][shard['command'].index('--dependency') + 1], 'afterok:103')
            self.assertEqual(finalizer['command'][finalizer['command'].index('--dependency') + 1], 'afterok:' + shard['job_id'])
            receipt = json.loads((development / 'results' / ('slurm_' + method + '_submission.json')).read_text())
            self.assertEqual(receipt['commands'][0][receipt['commands'][0].index('--dependency') + 1], 'afterok:' + finalizer['job_id'])
            self.assertIn(str(root / 'check.json'), receipt['commands'][0][-1])
        base_receipt = json.loads((development / 'results/slurm_submission.json').read_text())
        self.assertEqual([chunk['count'] for chunk in base_receipt['arrays']], [1001, 535])
        for command in base_receipt['commands']:
            self.assertEqual(command[command.index('--dependency') + 1], 'afterok:103')
            self.assertIn(str(directory / 'smoke.json'), command[-1])
        for command in self.accepted:
            self.assertEqual(command[command.index('--partition') + 1], 'l40sp')
            self.assertEqual(command[command.index('--time') + 1], '3-00:00:00')
            if '--array' in command:
                self.assertNotIn('%', command[command.index('--array') + 1])
        for receipt_file in (development / 'results').glob('*submission.json'):
            receipt = json.loads(receipt_file.read_text())
            self.assertEqual(receipt['output_dir'], str(development / 'results'))
            self.assertIsNone(receipt['max_parallel_gpus'])

    def test_queue_preview_does_not_create_calibration_or_submit(self):
        directory, plan, args = self.queue_fixture()
        args.submit = False
        with patch.object(s.subprocess, 'run') as process, \
             patch.object(s.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=2**50)), \
             contextlib.redirect_stdout(io.StringIO()):
            s.queue_development(args, repo=self.repo)
        process.assert_not_called()
        self.assertFalse((directory / 'calibration').exists())
        self.assertFalse((directory / 'development_l40sp').exists())

    def test_queue_rejects_unvalidated_partition_and_changed_launcher(self):
        directory, plan, args = self.queue_fixture()
        args.partitions = ['h200p', 'l40sp']
        with patch.object(s.subprocess, 'run') as process:
            with self.assertRaisesRegex(ValueError, 'smoke-tested partition'):
                s.queue_development(args, repo=self.repo)
            args.partitions = ['l40sp']
            launcher = self.repo / 'scripts/start_orion.py'
            launcher.write_text(launcher.read_text() + '\n# changed launcher\n')
            self.refresh()
            with self.assertRaisesRegex(ValueError, 'Calibration launcher changed'):
                s.queue_development(args, repo=self.repo)
        process.assert_not_called()

    def test_changed_calibration_inputs_cannot_replace_submitted_plan(self):
        directory, plan, args = self.queue_fixture()
        process = self.fake_slurm()
        with patch.object(s.subprocess, 'run', side_effect=process), \
             patch.object(s.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=2**50)), \
             contextlib.redirect_stdout(io.StringIO()):
            s.queue_development(args, repo=self.repo)
            count = len(self.accepted)
            self.sample_rows[0]['seed'] += 100
            self.write_samples(self.sample_rows)
            with self.assertRaisesRegex(ValueError, 'Existing experiment input differs'):
                s.queue_development(args, repo=self.repo)
        self.assertEqual(len(self.accepted), count)

    def test_sample_preparation_resolves_paths_without_downloading(self):
        directory, plan, args = self.queue_fixture()
        local = self.root / 'dog.mp4'
        local.write_bytes(b'fixture only')
        self.sample_rows.append({'prompt_id': 'local', 'prompt': 'A dog visits the local calibration scene.',
                                 'split': 'calibration', 'seed': 9, 'video_path': 'dog.mp4'})
        self.write_samples(self.sample_rows)
        with patch.object(s.subprocess, 'run') as process:
            rows = s.prepare_calibration_samples(self.samples, directory, [])
        process.assert_not_called()
        self.assertEqual(len(rows), 9)
        self.assertEqual(Path(rows[-1]['video_path']), local)
        self.assertEqual(Path(rows[0]['video_path']), directory / 'calibration/videos/0000.webm')
        self.assertFalse((directory / 'calibration').exists())

    def test_sample_preparation_rejects_prompt_and_identifier_leakage(self):
        directory, plan, args = self.queue_fixture()
        for field, value in [('prompt_en', '  A DOG explores calibration location number 0.  '),
                             ('prompt_id', 'calibration-0')]:
            held = self.root / 'held.json'
            held.write_text(json.dumps([{field: value}]))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'overlap'):
                s.prepare_calibration_samples(self.samples, directory, [held])
        held = self.root / 'held.jsonl'
        held.write_text(json.dumps({'prompt': self.sample_rows[-1]['prompt']}) + '\n')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            s.prepare_calibration_samples(self.samples, directory, [held])

    def test_sample_preparation_rejects_invalid_or_duplicate_assets(self):
        directory, plan, args = self.queue_fixture()
        cases = [('wrong split', {'split': 'test'}), ('blank prompt', {'prompt': ' '}),
                 ('negative seed', {'seed': -1}), ('boolean seed', {'seed': True}),
                 ('unsafe URL', {'video_url': 'http://example.invalid/dog.webm'}),
                 ('missing asset', {'video_url': None, 'video_path': 'missing.mp4'}),
                 ('duplicate prompt', {'prompt': self.sample_rows[1]['prompt']}),
                 ('duplicate ID', {'prompt_id': self.sample_rows[1]['prompt_id']}),
                 ('duplicate URL', {'video_url': self.sample_rows[1]['video_url']})]
        for name, change in cases:
            rows = [dict(row) for row in self.sample_rows]
            rows[0].update(change)
            self.write_samples(rows)
            with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                s.prepare_calibration_samples(self.samples, directory, [])

    def test_smoke_dependency_rejects_failed_prerequisites_before_query(self):
        directory, plan, args = self.queue_fixture()
        for name in ('env', 'model', 'smoke'):
            path = directory / (name + '.json')
            original = path.read_bytes() if path.exists() else None
            path.write_text(json.dumps({'status': 'failed'}))
            with self.subTest(name=name), patch.object(s.subprocess, 'run') as process:
                with self.assertRaisesRegex(ValueError, 'Prerequisite failed: ' + name):
                    s.valid_smoke_dependency(directory, plan)
                process.assert_not_called()
            if original is None:
                path.unlink()
            else:
                path.write_bytes(original)

    def test_smoke_dependency_rejects_stale_failed_and_unsatisfiable_jobs(self):
        directory, plan, args = self.queue_fixture()
        for output in ('', '103|FAILED|None\n', '103|PENDING|DependencyNeverSatisfied\n',
                       '999|RUNNING|None\n'):
            with self.subTest(output=output), patch.object(s.subprocess, 'run', return_value=types.SimpleNamespace(stdout=output)):
                with self.assertRaisesRegex(ValueError, 'no longer active'):
                    s.valid_smoke_dependency(directory, plan)
        with patch.object(s.subprocess, 'run', return_value=types.SimpleNamespace(stdout='103|PENDING|Dependency\n')):
            self.assertEqual(s.valid_smoke_dependency(directory, plan), 'afterok:103')
        (directory / 'smoke.json').write_text(json.dumps({'status': 'passed'}))
        with patch.object(s.subprocess, 'run') as process:
            self.assertIsNone(s.valid_smoke_dependency(directory, plan))
        process.assert_not_called()

    def test_calibration_receipt_blocks_uncertain_retry_and_changed_identity(self):
        receipt = self.root / 'receipt.json'
        command = ['sbatch', '--parsable', 'job.sh']
        with patch.object(s, 'run', return_value='unparseable response') as process:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected sbatch response'):
                s.submit_recorded_job(receipt, command, {'sample': 1})
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                s.submit_recorded_job(receipt, command, {'sample': 1})
            with self.assertRaisesRegex(ValueError, 'differs'):
                s.submit_recorded_job(receipt, command, {'sample': 2})
        self.assertEqual(process.call_count, 1)
        self.assertEqual(json.loads(receipt.read_text())['status'], 'uncertain')

if __name__=='__main__':unittest.main(verbosity=2)
