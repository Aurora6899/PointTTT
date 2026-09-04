import argparse
import os
import subprocess
import sys


CONFIG = 'configs/seg_scannet.yaml'
DATA_ROOT = 'data/scannet.npz'
TRAIN_DATA = os.path.join(DATA_ROOT, 'train')
TEST_DATA = os.path.join(DATA_ROOT, 'test')
TRAIN_LIST = os.path.join(DATA_ROOT, 'scannetv2_train_npz.txt')
VAL_LIST = os.path.join(DATA_ROOT, 'scannetv2_val_npz.txt')
TEST_LIST = os.path.join(DATA_ROOT, 'scannetv2_test_npz.txt')
TEST_SCENE_LIST = os.path.join(DATA_ROOT, 'scannetv2_test.txt')

TRAIN_LOGDIR = 'logs/scannet/hierarchical_pointttt_1cm'
VAL_VOTE_DIR = os.path.join(TRAIN_LOGDIR, 'val_votes')
VAL_PRED_DIR = os.path.join(TRAIN_LOGDIR, 'val_predictions')
TEST_VOTE_DIR = os.path.join(TRAIN_LOGDIR, 'test_votes')
TEST_PRED_DIR = os.path.join(TRAIN_LOGDIR, 'test_predictions')


parser = argparse.ArgumentParser(
    description='Train and evaluate PointTTT on ScanNet-20.')
parser.add_argument(
    '--run', choices=('train', 'validate', 'test', 'calc_iou'),
    default='train')
parser.add_argument(
    '--gpu', default='0,1',
    help='Comma-separated CUDA device ids; the default uses GPUs 0 and 1.')
parser.add_argument('--port', default='10001')
parser.add_argument(
    '--ckpt', default='',
    help='Checkpoint path. Defaults to the best checkpoint in TRAIN_LOGDIR.')
args = parser.parse_args()


def execute_command(cmd):
  print('Execute:\n' + ' '.join(cmd) + '\n', flush=True)
  subprocess.run(cmd, check=True)


def checkpoint_path():
  return args.ckpt or os.path.join(TRAIN_LOGDIR, 'best_model.pth')


def require_paths(*paths):
  missing = [path for path in paths if not os.path.exists(path)]
  if missing:
    raise FileNotFoundError(
        'Required ScanNet path(s) not found: ' + ', '.join(missing))


def solver_command():
  return [
      sys.executable,
      'segmentation.py',
      '--config', CONFIG,
      'SOLVER.gpu', args.gpu + ',',
      'SOLVER.dist_url', 'tcp://localhost:' + args.port,
  ]


def train():
  require_paths(CONFIG, TRAIN_DATA, TRAIN_LIST, VAL_LIST)
  execute_command(solver_command())


def generate_predictions(vote_dir, pred_dir, filelist, data_dir):
  execute_command([
      sys.executable,
      'tools/seg_scannet.py',
      '--run', 'generate_output_seg',
      '--path_in', data_dir,
      '--path_pred', vote_dir,
      '--path_out', pred_dir,
      '--filelist', filelist,
  ])


def validate():
  # OctFormer reports validation results with 120 augmented votes.
  require_paths(CONFIG, TRAIN_DATA, VAL_LIST, checkpoint_path())
  cmd = solver_command() + [
      'LOSS.mask', '-255',
      'SOLVER.run', 'evaluate',
      'SOLVER.eval_epoch', '120',
      'SOLVER.ckpt', checkpoint_path(),
      'SOLVER.logdir', VAL_VOTE_DIR,
      'DATA.test.location', TRAIN_DATA,
      'DATA.test.filelist', VAL_LIST,
      'DATA.test.batch_size', '1',
      'DATA.test.shuffle', 'False',
      'DATA.test.distort', 'True',
  ]
  execute_command(cmd)
  generate_predictions(VAL_VOTE_DIR, VAL_PRED_DIR, VAL_LIST, TRAIN_DATA)
  calc_iou()


def test():
  # OctFormer produces ScanNet test-server predictions with 72 augmented votes.
  require_paths(
      CONFIG, TEST_DATA, TEST_LIST, TEST_SCENE_LIST, checkpoint_path())
  cmd = solver_command() + [
      'LOSS.mask', '-255',
      'SOLVER.run', 'evaluate',
      'SOLVER.eval_epoch', '72',
      'SOLVER.ckpt', checkpoint_path(),
      'SOLVER.logdir', TEST_VOTE_DIR,
      'DATA.test.location', TEST_DATA,
      'DATA.test.filelist', TEST_LIST,
      'DATA.test.batch_size', '1',
      'DATA.test.shuffle', 'False',
      'DATA.test.distort', 'True',
  ]
  execute_command(cmd)
  generate_predictions(
      TEST_VOTE_DIR, TEST_PRED_DIR, TEST_SCENE_LIST, TEST_DATA)


def calc_iou():
  require_paths(TRAIN_DATA, VAL_PRED_DIR)
  execute_command([
      sys.executable,
      'tools/seg_scannet.py',
      '--run', 'calc_iou',
      '--path_in', TRAIN_DATA,
      '--path_pred', VAL_PRED_DIR,
  ])


if __name__ == '__main__':
  globals()[args.run]()
