import torch
from typing import Optional, Union

class MultiKeyLUT:
    """扩展的多序列化方法KeyLUT"""
    
    def __init__(self):

        r256 = torch.arange(256, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device('cpu')
        

        self._z_encode = {device: (self.xyz2key_z_order(r256, zero, zero, 8),
                                   self.xyz2key_z_order(zero, r256, zero, 8),
                                   self.xyz2key_z_order(zero, zero, r256, 8))}
        
        # 2. Trans Z-order
        self._trans_z_encode = {device: (self.xyz2key_trans_z_order(r256, zero, zero, 8),
                                        self.xyz2key_trans_z_order(zero, r256, zero, 8),
                                        self.xyz2key_trans_z_order(zero, zero, r256, 8))}
        

        self._hilbert_encode = {device: (self.xyz2key_hilbert_simple(r256, zero, zero, 8),
                                        self.xyz2key_hilbert_simple(zero, r256, zero, 8),
                                        self.xyz2key_hilbert_simple(zero, zero, r256, 8))}
        
        # 4. Trans Hilbert
        self._trans_hilbert_encode = {device: (self.xyz2key_trans_hilbert_simple(r256, zero, zero, 8),
                                              self.xyz2key_trans_hilbert_simple(zero, r256, zero, 8),
                                              self.xyz2key_trans_hilbert_simple(zero, zero, r256, 8))}

    def get_encode_lut(self, method='z_order', device=torch.device('cpu')):
        """获取编码查找表"""
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


    def xyz2key_z_order(self, x, y, z, depth):
        """原始Z-order编码: xyz位交错"""
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((x & mask) << (2 * i + 2)) |
                         ((y & mask) << (2 * i + 1)) |
                         ((z & mask) << (2 * i + 0)))
        return key

    # === Trans Z-order ===
    def xyz2key_trans_z_order(self, x, y, z, depth):
        """Trans Z-order编码: zyx位交错"""
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((z & mask) << (2 * i + 2)) |
                         ((y & mask) << (2 * i + 1)) |
                         ((x & mask) << (2 * i + 0)))
        return key


    def xyz2key_hilbert_simple(self, x, y, z, depth):
        """简化版3D Hilbert曲线编码 - 避免索引问题"""
        key = torch.zeros_like(x)
        for i in range(depth):

            x_bit = (x >> i) & 1
            y_bit = (y >> i) & 1
            z_bit = (z >> i) & 1
            

            hilbert_code = self._simple_hilbert_transform(x_bit, y_bit, z_bit, i)
            key = key | (hilbert_code << (3 * i))
        return key

    def _simple_hilbert_transform(self, x, y, z, level):
        """简化的3D Hilbert变换 - 使用数学运算而非查找表"""

        gray_x = x ^ (x >> 1)
        gray_y = y ^ (y >> 1) 
        gray_z = z ^ (z >> 1)
        

        if level % 3 == 0:
            return gray_x * 4 + gray_y * 2 + gray_z
        elif level % 3 == 1:
            return gray_z * 4 + gray_x * 2 + gray_y
        else:
            return gray_y * 4 + gray_z * 2 + gray_x

    # === Trans Hilbert ===
    def xyz2key_trans_hilbert_simple(self, x, y, z, depth):
        """Trans Hilbert: 坐标转置后应用简化Hilbert"""

        return self.xyz2key_hilbert_simple(z, x, y, depth)



_multi_key_lut = MultiKeyLUT()


def multi_xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                  b: Optional[Union[torch.Tensor, int]] = None, 
                  depth: int = 16, method: str = 'z_order'):
    """多种序列化方法的统一接口"""
    
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
