import torch
from typing import Optional, Union

class MultiKeyLUT:
    """Lookup tables for the serialization modes used by PointTTT ASR."""
    
    def __init__(self):
        # Base lookup tensors.
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device('cpu')
        
        # 1. Z-order.
        self._z_encode = {device: (self.xyz2key_z_order(r256, zero, zero, 8),
                                   self.xyz2key_z_order(zero, r256, zero, 8),
                                   self.xyz2key_z_order(zero, zero, r256, 8))}
        self._z_decode = {device: self.key2xyz_z_order(r512, 9)}
        
        # 2. Trans Z-order
        self._trans_z_encode = {device: (self.xyz2key_trans_z_order(r256, zero, zero, 8),
                                        self.xyz2key_trans_z_order(zero, r256, zero, 8),
                                        self.xyz2key_trans_z_order(zero, zero, r256, 8))}
        self._trans_z_decode = {device: self.key2xyz_trans_z_order(r512, 9)}
        
        # 3. Hilbert, implemented with a lightweight approximation.
        self._hilbert_encode = {device: (self.xyz2key_hilbert_simple(r256, zero, zero, 8),
                                        self.xyz2key_hilbert_simple(zero, r256, zero, 8),
                                        self.xyz2key_hilbert_simple(zero, zero, r256, 8))}
        self._hilbert_decode = {device: self.key2xyz_hilbert_simple(r512, 9)}
        
        # 4. Trans Hilbert
        self._trans_hilbert_encode = {device: (self.xyz2key_trans_hilbert_simple(r256, zero, zero, 8),
                                              self.xyz2key_trans_hilbert_simple(zero, r256, zero, 8),
                                              self.xyz2key_trans_hilbert_simple(zero, zero, r256, 8))}
        self._trans_hilbert_decode = {device: self.key2xyz_trans_hilbert_simple(r512, 9)}

    def get_encode_lut(self, method='z_order', device=torch.device('cpu')):
        """Return the encode lookup table for one serialization method."""
        lut_dict = {
            'z_order': self._z_encode,
            'trans_z': self._trans_z_encode, 
            'hilbert': self._hilbert_encode,
            'trans_hilbert': self._trans_hilbert_encode
        }
        
        if device not in lut_dict[method]:
            cpu = torch.device('cpu')
            lut_dict[method][device] = tuple(e.to(device) for e in lut_dict[method][cpu])
        
        return lut_dict[method][device]

    def get_decode_lut(self, method='z_order', device=torch.device('cpu')):
        """Return the decode lookup table for one serialization method."""
        lut_dict = {
            'z_order': self._z_decode,
            'trans_z': self._trans_z_decode,
            'hilbert': self._hilbert_decode, 
            'trans_hilbert': self._trans_hilbert_decode
        }
        
        if device not in lut_dict[method]:
            cpu = torch.device('cpu')
            lut_dict[method][device] = tuple(e.to(device) for e in lut_dict[method][cpu])
        
        return lut_dict[method][device]

    # === Z-order ===
    def xyz2key_z_order(self, x, y, z, depth):
        """Encode coordinates with xyz bit interleaving."""
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((x & mask) << (2 * i + 2)) |
                         ((y & mask) << (2 * i + 1)) |
                         ((z & mask) << (2 * i + 0)))
        return key

    def key2xyz_z_order(self, key, depth):
        """Decode a Z-order key."""
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z

    # === Trans Z-order ===
    def xyz2key_trans_z_order(self, x, y, z, depth):
        """Encode coordinates with zyx bit interleaving."""
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((z & mask) << (2 * i + 2)) |
                         ((y & mask) << (2 * i + 1)) |
                         ((x & mask) << (2 * i + 0)))
        return key

    def key2xyz_trans_z_order(self, key, depth):
        """Decode a transposed Z-order key."""
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            z = z | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            x = x | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z

    # === Lightweight Hilbert approximation ===
    def xyz2key_hilbert_simple(self, x, y, z, depth):
        """Encode coordinates with a lightweight 3D Hilbert approximation."""
        key = torch.zeros_like(x)
        for i in range(depth):
            # Extract the coordinate bit at the current level.
            x_bit = (x >> i) & 1
            y_bit = (y >> i) & 1
            z_bit = (z >> i) & 1
            
            # Approximate the Hilbert transform without large lookup tables.
            hilbert_code = self._simple_hilbert_transform(x_bit, y_bit, z_bit, i)
            key = key | (hilbert_code << (3 * i))
        return key

    def key2xyz_hilbert_simple(self, key, depth):
        """Decode the lightweight 3D Hilbert approximation."""
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            hilbert_code = (key >> (3 * i)) & 7
            x_bit, y_bit, z_bit = self._simple_hilbert_inverse(hilbert_code, i)
            x = x | (x_bit << i)
            y = y | (y_bit << i)
            z = z | (z_bit << i)
        return x, y, z

    def _simple_hilbert_transform(self, x, y, z, level):
        """Approximate 3D Hilbert transform using Gray-code operations."""
        gray_x = x ^ (x >> 1)
        gray_y = y ^ (y >> 1) 
        gray_z = z ^ (z >> 1)
        
        # Rotate the coordinate bit order across levels.
        if level % 3 == 0:
            return gray_x * 4 + gray_y * 2 + gray_z
        elif level % 3 == 1:
            return gray_z * 4 + gray_x * 2 + gray_y
        else:
            return gray_y * 4 + gray_z * 2 + gray_x

    def _simple_hilbert_inverse(self, hilbert_code, level):
        """Inverse of the lightweight 3D Hilbert approximation."""
        if level % 3 == 0:
            gray_x = (hilbert_code >> 2) & 1
            gray_y = (hilbert_code >> 1) & 1
            gray_z = hilbert_code & 1
        elif level % 3 == 1:
            gray_z = (hilbert_code >> 2) & 1
            gray_x = (hilbert_code >> 1) & 1
            gray_y = hilbert_code & 1
        else:
            gray_y = (hilbert_code >> 2) & 1
            gray_z = (hilbert_code >> 1) & 1
            gray_x = hilbert_code & 1
        
        # Inverse Gray-code transform.
        x = gray_x ^ (gray_x >> 1)
        y = gray_y ^ (gray_y >> 1)
        z = gray_z ^ (gray_z >> 1)
        
        return x, y, z

    # === Trans Hilbert ===
    def xyz2key_trans_hilbert_simple(self, x, y, z, depth):
        """Apply the lightweight Hilbert transform after coordinate transposition."""
        return self.xyz2key_hilbert_simple(z, x, y, depth)

    def key2xyz_trans_hilbert_simple(self, key, depth):
        """Decode a transposed Hilbert key."""
        z, x, y = self.key2xyz_hilbert_simple(key, depth)
        return x, y, z


# Global lookup-table instance.
_multi_key_lut = MultiKeyLUT()


def multi_xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                  b: Optional[Union[torch.Tensor, int]] = None, 
                  depth: int = 16, method: str = 'z_order'):
    """Unified coordinate-to-key interface for PointTTT serialization modes."""
    
    EX, EY, EZ = _multi_key_lut.get_encode_lut(method, x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    
    if depth > 8:
        mask = (1 << (depth-8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key


def multi_key2xyz(key: torch.Tensor, depth: int = 16, method: str = 'z_order'):
    """Unified key-to-coordinate interface for PointTTT serialization modes."""
    
    DX, DY, DZ = _multi_key_lut.get_decode_lut(method, key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b
