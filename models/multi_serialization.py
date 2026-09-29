import itertools
from typing import Dict, Optional, Tuple, Union

import torch


# A canonical axis suffix describes the coordinates passed to the base encoder,
# e.g. ``morton_zxy`` encodes ``(z, x, y)``.
AXIS_PERMUTATIONS: Dict[str, Tuple[int, int, int]] = {
    ''.join(name): tuple(perm)
    for name, perm in zip(
        itertools.permutations('xyz'), itertools.permutations(range(3)))
}

METHOD_SPECS: Dict[str, Tuple[str, Tuple[int, int, int]]] = {
    'morton_' + suffix: ('morton', permutation)
    for suffix, permutation in AXIS_PERMUTATIONS.items()
}
METHOD_SPECS.update({
    'hilbert_' + suffix: ('hilbert', permutation)
    for suffix, permutation in AXIS_PERMUTATIONS.items()
})

# Paper Axis12 names for the four canonical traversal variants.
METHOD_SPECS.update({
    'z_order': METHOD_SPECS['morton_xyz'],
    'trans_z': METHOD_SPECS['morton_zyx'],
    'hilbert': METHOD_SPECS['hilbert_xyz'],
    'trans_hilbert': METHOD_SPECS['hilbert_zxy'],
})

AXIS12_METHODS = (
    'z_order',
    'morton_xzy',
    'morton_yxz',
    'morton_yzx',
    'morton_zxy',
    'trans_z',
    'hilbert',
    'hilbert_xzy',
    'hilbert_yxz',
    'hilbert_yzx',
    'trans_hilbert',
    'hilbert_zyx',
)
class MultiKeyLUT:
    """Lookup tables for Morton and simplified-Hilbert axis permutations."""

    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device('cpu')

        self._encode = {}
        self._decode = {}
        for method in METHOD_SPECS:
            self._encode[method] = {
                device: (
                    self.xyz2key(r256, zero, zero, 8, method),
                    self.xyz2key(zero, r256, zero, 8, method),
                    self.xyz2key(zero, zero, r256, 8, method),
                )
            }
            self._decode[method] = {
                device: self.key2xyz(r512, 9, method)
            }

    @staticmethod
    def _permute(x, y, z, permutation):
        coordinates = (x, y, z)
        return tuple(coordinates[index] for index in permutation)

    @staticmethod
    def _inverse_permute(a, b, c, permutation):
        permuted = (a, b, c)
        coordinates = [None, None, None]
        for output_index, input_index in enumerate(permutation):
            coordinates[input_index] = permuted[output_index]
        return tuple(coordinates)

    def get_encode_lut(self, method='z_order', device=torch.device('cpu')):
        if method not in self._encode:
            raise ValueError('Unsupported serialization method: ' + str(method))
        if device not in self._encode[method]:
            cpu = torch.device('cpu')
            self._encode[method][device] = tuple(
                item.to(device) for item in self._encode[method][cpu])
        return self._encode[method][device]

    def get_decode_lut(self, method='z_order', device=torch.device('cpu')):
        if method not in self._decode:
            raise ValueError('Unsupported serialization method: ' + str(method))
        if device not in self._decode[method]:
            cpu = torch.device('cpu')
            self._decode[method][device] = tuple(
                item.to(device) for item in self._decode[method][cpu])
        return self._decode[method][device]

    def xyz2key(self, x, y, z, depth, method):
        family, permutation = METHOD_SPECS[method]
        a, b, c = self._permute(x, y, z, permutation)
        if family == 'morton':
            return self.xyz2key_morton(a, b, c, depth)
        return self.xyz2key_hilbert_simple(a, b, c, depth)

    def key2xyz(self, key, depth, method):
        family, permutation = METHOD_SPECS[method]
        if family == 'morton':
            a, b, c = self.key2xyz_morton(key, depth)
        else:
            a, b, c = self.key2xyz_hilbert_simple(key, depth)
        return self._inverse_permute(a, b, c, permutation)

    @staticmethod
    def xyz2key_morton(x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((x & mask) << (2 * i + 2)) |
                   ((y & mask) << (2 * i + 1)) |
                   ((z & mask) << (2 * i + 0)))
        return key

    @staticmethod
    def key2xyz_morton(key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z

    def xyz2key_hilbert_simple(self, x, y, z, depth):
        """The project's existing level-cycled, simplified Hilbert encoder."""
        key = torch.zeros_like(x)
        for i in range(depth):
            x_bit = (x >> i) & 1
            y_bit = (y >> i) & 1
            z_bit = (z >> i) & 1
            code = self._simple_hilbert_transform(x_bit, y_bit, z_bit, i)
            key = key | (code << (3 * i))
        return key

    def key2xyz_hilbert_simple(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            code = (key >> (3 * i)) & 7
            x_bit, y_bit, z_bit = self._simple_hilbert_inverse(code, i)
            x = x | (x_bit << i)
            y = y | (y_bit << i)
            z = z | (z_bit << i)
        return x, y, z

    @staticmethod
    def _simple_hilbert_transform(x, y, z, level):
        gray_x = x ^ (x >> 1)
        gray_y = y ^ (y >> 1)
        gray_z = z ^ (z >> 1)
        if level % 3 == 0:
            return gray_x * 4 + gray_y * 2 + gray_z
        if level % 3 == 1:
            return gray_z * 4 + gray_x * 2 + gray_y
        return gray_y * 4 + gray_z * 2 + gray_x

    @staticmethod
    def _simple_hilbert_inverse(code, level):
        if level % 3 == 0:
            gray_x = (code >> 2) & 1
            gray_y = (code >> 1) & 1
            gray_z = code & 1
        elif level % 3 == 1:
            gray_z = (code >> 2) & 1
            gray_x = (code >> 1) & 1
            gray_y = code & 1
        else:
            gray_y = (code >> 2) & 1
            gray_z = (code >> 1) & 1
            gray_x = code & 1
        # Each value is one bit, so inverse Gray is identical here. Keep the
        # operation explicit to preserve the original implementation.
        x = gray_x ^ (gray_x >> 1)
        y = gray_y ^ (gray_y >> 1)
        z = gray_z ^ (gray_z >> 1)
        return x, y, z

_multi_key_lut = MultiKeyLUT()


def multi_xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                  b: Optional[Union[torch.Tensor, int]] = None,
                  depth: int = 16, method: str = 'z_order'):
    """Encodes quantized octree coordinates with a registered serialization."""
    x, y, z = x.long(), y.long(), z.long()
    family, _ = METHOD_SPECS[method]
    if family == 'hilbert':
        # The simplified Hilbert encoder changes its axis phase with the
        # absolute bit level. Splitting at bit 8 would restart that phase, so
        # encode all levels directly.
        key = _multi_key_lut.xyz2key(x, y, z, depth, method)
    else:
        EX, EY, EZ = _multi_key_lut.get_encode_lut(method, x.device)
        mask = 255 if depth > 8 else (1 << depth) - 1
        key = EX[x & mask] | EY[y & mask] | EZ[z & mask]

        if depth > 8:
            mask = (1 << (depth - 8)) - 1
            key16 = (EX[(x >> 8) & mask] | EY[(y >> 8) & mask] |
                     EZ[(z >> 8) & mask])
            key = key16 << 24 | key

    if b is not None:
        b = torch.as_tensor(b, device=key.device, dtype=torch.int64)
        key = b << 48 | key
    return key


def multi_key2xyz(key: torch.Tensor, depth: int = 16,
                  method: str = 'z_order'):
    """Decodes keys generated by :func:`multi_xyz2key`."""
    b = key >> 48
    key = key & ((1 << 48) - 1)
    family, _ = METHOD_SPECS[method]
    if family == 'hilbert':
        x, y, z = _multi_key_lut.key2xyz(key, depth, method)
        return x, y, z, b

    DX, DY, DZ = _multi_key_lut.get_decode_lut(method, key.device)
    x = torch.zeros_like(key)
    y = torch.zeros_like(key)
    z = torch.zeros_like(key)
    n = (depth + 2) // 3
    for i in range(n):
        chunk = key >> (i * 9) & 511
        x = x | (DX[chunk] << (i * 3))
        y = y | (DY[chunk] << (i * 3))
        z = z | (DZ[chunk] << (i * 3))
    return x, y, z, b
