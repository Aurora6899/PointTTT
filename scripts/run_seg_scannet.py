import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='train')
parser.add_argument('--alias', type=str, required=False, default='scannet')
parser.add_argument('--gpu', type=str, required=False, default='0')
parser.add_argument('--port', type=str, required=False, default='10001')
parser.add_argument('--ckpt', type=str, required=False, default='')
args = parser.parse_args()


def execute_command(cmds):
  cmd = ' '.join(cmds)
  print('Execute: \n' + cmd + '\n')
  os.system(cmd)


def train():
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.alias  {}'.format(args.alias),
      'SOLVER.dist_url tcp://localhost:{}'.format(args.port),]
  execute_command(cmds)


def test():
  ckpt = ('logs/scannet/pointttt_{}/best_model.pth'.format(args.alias)
          if args.ckpt == '' else args.ckpt)
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'LOSS.mask -255',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.run evaluate',
      'SOLVER.eval_epoch 72',
      'SOLVER.alias test_{}'.format(args.alias),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.test.batch_size 1',
      'DATA.test.location data/scannet.npz/test',
      'DATA.test.filelist data/scannet.npz/scannetv2_test_npz.txt',
      'DATA.test.distort True',]
  execute_command(cmds)

  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_pred logs/scannet/pointttt_test_{}'.format(args.alias),
      '--path_out logs/scannet/pointttt_test_seg_{}'.format(args.alias),
      '--filelist data/scannet.npz/scannetv2_test.txt',]
  execute_command(cmds)


def validate():
  ckpt = args.ckpt if args.ckpt else 'logs/scannet/pointttt_{}/best_model.pth'.format(args.alias)
  cmds = [
      'python segmentation.py',
      '--config configs/seg_scannet.yaml',
      'LOSS.mask -255',
      'SOLVER.gpu  {},'.format(args.gpu),
      'SOLVER.run evaluate',
      'SOLVER.eval_epoch 120',
      'SOLVER.alias val_{}'.format(args.alias),
      'SOLVER.ckpt {}'.format(ckpt),
      'DATA.test.batch_size 1',
      'DATA.test.distort True',]
  execute_command(cmds)

  cmds = [
      'python tools/seg_scannet.py',
      '--run generate_output_seg',
      '--path_pred logs/scannet/pointttt_val_{}'.format(args.alias),
      '--path_out logs/scannet/pointttt_val_seg_{}'.format(args.alias),
      '--filelist data/scannet.npz/scannetv2_val_npz.txt',]
  execute_command(cmds)

  calc_iou()


def calc_iou():
  cmds = [
      'python tools/seg_scannet.py',
      '--run calc_iou',
      '--path_in data/scannet.npz/train',
      '--path_pred logs/scannet/pointttt_val_seg_{}'.format(args.alias),]
  execute_command(cmds)


RUNNERS = {
    'train': train,
    'test': test,
    'validate': validate,
    'calc_iou': calc_iou,
}


if __name__ == '__main__':
  RUNNERS[args.run]()
