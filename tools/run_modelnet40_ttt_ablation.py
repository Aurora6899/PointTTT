#!/usr/bin/env python3
"""Run the requested ModelNet40 TTT ablation on two independent GPUs."""

import csv
import json
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / 'configs/cls_modelnet40_ttt_ablation.yaml'
OUTPUT_ROOT = REPO_ROOT / 'logs/modelnet40/ttt_update_ablation'
SUMMARY = OUTPUT_ROOT / 'summary.csv'
JOBS = [
    dict(name='no_update_retrained', base_lr=1.0, update=False),
    dict(name='lr_1p0', base_lr=1.0, update=True),
    dict(name='lr_0p1', base_lr=0.1, update=True),
    dict(name='lr_0p2', base_lr=0.2, update=True),
    dict(name='lr_0p5', base_lr=0.5, update=True),
    dict(name='lr_2p0', base_lr=2.0, update=True),
    dict(name='lr_3p0', base_lr=3.0, update=True),
]


def bool_text(value):
  return 'True' if value else 'False'


def stream_command(command, log_path, prefix):
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open('w', buffering=1) as log_file:
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    print('[%s] PID=%d' % (prefix, process.pid), flush=True)
    for line in process.stdout:
      log_file.write(line)
      clean = line.strip()
      if ('[TEST] acc' in clean or 'Best model at epoch' in clean or
          re.search(r'=> Epoch: (1|25|50|75|100|125|150),', clean)):
        print('[%s] %s' % (prefix, clean), flush=True)
    return process.wait()


def parse_best(path):
  text = path.read_text() if path.is_file() else ''
  matches = re.findall(
      r'epoch:\s*(\d+),\s*test/accu:\s*([0-9.eE+-]+)', text)
  if not matches:
    raise RuntimeError('Could not parse %s' % path)
  epoch, accuracy = max(matches, key=lambda item: float(item[1]))
  return int(epoch), float(accuracy)


def profile(job, gpu, checkpoint, output_dir):
  command = [
      sys.executable, 'tools/profile_modelnet40_ttt_update.py',
      '--checkpoint', str(checkpoint), '--output', str(output_dir / 'profile.json'),
      '--gpu', str(gpu), '--base-lr', str(job['base_lr']),
      '--update', 'on' if job['update'] else 'off',
      '--warmup', '50', '--repeat', '200']
  code = stream_command(command, output_dir / 'profile_console.log',
                        job['name'] + '/profile')
  if code != 0:
    raise RuntimeError('%s profile exited with %d' % (job['name'], code))
  return json.loads((output_dir / 'profile.json').read_text())


def train(job, gpu):
  output_dir = OUTPUT_ROOT / job['name']
  command = [
      sys.executable, 'classification.py', '--config', str(CONFIG),
      'SOLVER.gpu', '%d,' % gpu,
      'SOLVER.logdir', str(output_dir),
      'SOLVER.max_epoch', '150',
      'MODEL.ttt_base_lr', str(job['base_lr']),
      'MODEL.ttt_update_train', bool_text(job['update']),
      'MODEL.ttt_update_test', bool_text(job['update'])]
  print('[%s] START gpu=%d base_lr=%s update=%s' %
        (job['name'], gpu, job['base_lr'], job['update']), flush=True)
  code = stream_command(command, output_dir / 'console.log', job['name'])
  if code != 0:
    raise RuntimeError('%s training exited with %d' % (job['name'], code))
  epoch, oa = parse_best(output_dir / 'best_model.txt')
  checkpoint = output_dir / 'best_model.pth'
  metrics = profile(job, gpu, checkpoint, output_dir)
  result = dict(job)
  result.update(status='completed', gpu=gpu, best_epoch=epoch, oa=oa,
                latency_ms=metrics['latency_ms'],
                peak_memory_mib=metrics['peak_memory_mib'],
                checkpoint=str(checkpoint))
  print('[%s] COMPLETE best_epoch=%d OA=%.6f latency=%.3fms peak=%.1fMiB' %
        (job['name'], epoch, oa, metrics['latency_ms'],
         metrics['peak_memory_mib']), flush=True)
  return result


def evaluate_same_checkpoint_test_off(gpu):
  source = OUTPUT_ROOT / 'lr_1p0/best_model.pth'
  output_dir = OUTPUT_ROOT / 'lr_1p0_same_checkpoint_test_off'
  command = [
      sys.executable, 'classification.py', '--config', str(CONFIG),
      'SOLVER.gpu', '%d,' % gpu,
      'SOLVER.run', 'test', 'SOLVER.ckpt', str(source),
      'SOLVER.logdir', str(output_dir),
      'MODEL.ttt_base_lr', '1.0',
      'MODEL.ttt_update_train', 'True',
      'MODEL.ttt_update_test', 'False']
  print('[same_checkpoint_test_off] START gpu=%d' % gpu, flush=True)
  code = stream_command(command, output_dir / 'console.log',
                        'same_checkpoint_test_off')
  if code != 0:
    raise RuntimeError('same-checkpoint test-off exited with %d' % code)
  log_text = (output_dir / 'log.csv').read_text()
  matches = re.findall(r'test/accu:\s*([0-9.eE+-]+)', log_text)
  if not matches:
    raise RuntimeError('Could not parse test-off OA')
  job = dict(name='lr_1p0_same_checkpoint_test_off', base_lr=1.0,
             update=False)
  metrics = profile(job, gpu, source, output_dir)
  result = dict(job)
  result.update(status='completed', gpu=gpu, best_epoch='',
                oa=float(matches[-1]), latency_ms=metrics['latency_ms'],
                peak_memory_mib=metrics['peak_memory_mib'], checkpoint=str(source))
  print('[same_checkpoint_test_off] COMPLETE OA=%.6f latency=%.3fms peak=%.1fMiB' %
        (result['oa'], result['latency_ms'], result['peak_memory_mib']),
        flush=True)
  return result


def write_summary(results):
  fields = ['name', 'status', 'gpu', 'base_lr', 'update', 'best_epoch',
            'oa', 'latency_ms', 'peak_memory_mib', 'checkpoint', 'error']
  SUMMARY.parent.mkdir(parents=True, exist_ok=True)
  with SUMMARY.open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for result in results:
      writer.writerow({field: result.get(field, '') for field in fields})


def main():
  OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
  pending = queue.Queue()
  for job in JOBS:
    pending.put(job)
  results = []
  lock = threading.Lock()

  def worker(gpu):
    while True:
      try:
        job = pending.get_nowait()
      except queue.Empty:
        return
      try:
        result = train(job, gpu)
      except Exception as error:
        result = dict(job, status='failed', gpu=gpu, error=str(error))
        print('[%s] FAILED: %s' % (job['name'], error), flush=True)
        with lock:
          results.append(result)
          write_summary(results)
        return
      with lock:
        results.append(result)
        write_summary(results)
      pending.task_done()

  workers = [threading.Thread(target=worker, args=(gpu,), daemon=False)
             for gpu in (0, 1)]
  for worker_thread in workers:
    worker_thread.start()
  for worker_thread in workers:
    worker_thread.join()

  if all(result.get('status') == 'completed' for result in results) and \
      len(results) == len(JOBS):
    try:
      results.append(evaluate_same_checkpoint_test_off(0))
    except Exception as error:
      results.append(dict(
          name='lr_1p0_same_checkpoint_test_off', status='failed', gpu=0,
          base_lr=1.0, update=False, error=str(error)))
      print('[same_checkpoint_test_off] FAILED: %s' % error, flush=True)
  write_summary(results)
  failures = [result for result in results if result.get('status') != 'completed']
  print('SUMMARY %s completed=%d failed=%d' %
        (SUMMARY, len(results) - len(failures), len(failures)), flush=True)
  raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
  main()
