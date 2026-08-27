"""Launch MMDetection3D training after registering the PointTTT detector.

All MMDetection3D command-line options are passed through unchanged.  The
launcher first uses the active environment; as a convenience it can also find
the sibling OctFormer checkout specified by ``MMDET3D_ROOT``.
"""

import os
import runpy
import sys
from pathlib import Path


def _find_mmdet3d():
    try:
        import mmdet3d
        return mmdet3d
    except (ImportError, AssertionError) as first_error:
        candidates = []
        configured = os.environ.get('MMDET3D_ROOT')
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(
            Path(__file__).resolve().parents[1]
            / 'octformer-master' / 'mmdetection3d')

        for root in candidates:
            if (root / 'mmdet3d' / '__init__.py').is_file():
                sys.path.insert(0, str(root))
                try:
                    import mmdet3d
                    return mmdet3d
                except (ImportError, AssertionError):
                    sys.path.pop(0)
        raise RuntimeError(
            'MMDetection3D v1.0.0rc5 is not usable in the active environment. '
            'Activate the mamba environment and install the optional detection '
            'requirements listed in requirements-detection.txt.') from first_error


def _patch_legacy_mmseg_api():
    """Bridge the local MMDetection3D checkout to mmsegmentation 0.29.x."""
    import mmseg.utils
    if not hasattr(mmseg.utils, 'add_prefix'):
        from mmseg.core import add_prefix
        mmseg.utils.add_prefix = add_prefix


def _patch_mmcv_ddp_api():
    """Make MMCV 1.7 MMDistributedDataParallel work with PyTorch 2.2.

    MMCV 1.7.2's evaluation forward accesses PyTorch DDP's historical
    ``_use_replicated_tensor_module`` attribute.  PyTorch 2.2 removed that
    private attribute, although the normal ``module`` path is still valid.
    Keep the legacy behavior when the attribute exists and otherwise select
    the wrapped module directly.  The patch is local to the detection launcher
    and does not modify the active environment or other project entry points.
    """
    import torch
    from mmcv.parallel import MMDistributedDataParallel

    torch_ddp = torch.nn.parallel.DistributedDataParallel
    if hasattr(torch_ddp, '_use_replicated_tensor_module'):
        return
    if getattr(MMDistributedDataParallel,
               '_pointttt_torch22_compatible', False):
        return

    def _run_ddp_forward(self, *inputs, **kwargs):
        use_replicated = getattr(
            self, '_use_replicated_tensor_module', False)
        replicated = getattr(self, '_replicated_tensor_module', None)
        module_to_run = replicated if use_replicated and replicated is not None \
            else self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])
        return module_to_run(*inputs, **kwargs)

    MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward
    MMDistributedDataParallel._pointttt_torch22_compatible = True


def main():
    mmdet3d = _find_mmdet3d()
    _patch_legacy_mmseg_api()
    _patch_mmcv_ddp_api()

    # Importing this module registers both custom classes with MMDetection3D.
    from models import PointTTTdet  # noqa: F401

    package_root = Path(mmdet3d.__file__).resolve().parent.parent
    train_script = package_root / 'tools' / 'train.py'
    if not train_script.is_file():
        raise FileNotFoundError(
            f'Cannot find the MMDetection3D training entry at {train_script}')
    runpy.run_path(str(train_script), run_name='__main__')


if __name__ == '__main__':
    main()
