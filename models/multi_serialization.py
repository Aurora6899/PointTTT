import torch
from typing import Optional, Union

class MultiKeyLUT:
    """扩展的多序列化方法KeyLUT"""
    
    def __init__(self):
        # 复用原有的基础设置
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device('cpu')
        
        # 1. Z-order (原始方法)
        self._z_encode = {device: (self.xyz2key_z_order(r256, zero, zero, 8),
                                   self.xyz2key_z_order(zero, r256, zero, 8),
                                   self.xyz2key_z_order(zero, zero, r256, 8))}
        self._z_decode = {device: self.key2xyz_z_order(r512, 9)}
        
        # 2. Trans Z-order
        self._trans_z_encode = {device: (self.xyz2key_trans_z_order(r256, zero, zero, 8),
                                        self.xyz2key_trans_z_order(zero, r256, zero, 8),
                                        self.xyz2key_trans_z_order(zero, zero, r256, 8))}
        self._trans_z_decode = {device: self.key2xyz_trans_z_order(r512, 9)}
        
        # 3. Hilbert - 先使用简化版本
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

    def get_decode_lut(self, method='z_order', device=torch.device('cpu')):
        """获取解码查找表"""
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

    # === Z-order (原始方法) ===
    def xyz2key_z_order(self, x, y, z, depth):
        """原始Z-order编码: xyz位交错"""
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((x & mask) << (2 * i + 2)) |  # x最高位
                         ((y & mask) << (2 * i + 1)) |  # y中间位
                         ((z & mask) << (2 * i + 0)))   # z最低位
        return key

    def key2xyz_z_order(self, key, depth):
        """原始Z-order解码"""
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
        """Trans Z-order编码: zyx位交错"""
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((z & mask) << (2 * i + 2)) |  # z最高位
                         ((y & mask) << (2 * i + 1)) |  # y中间位
                         ((x & mask) << (2 * i + 0)))   # x最低位
        return key

    def key2xyz_trans_z_order(self, key, depth):
        """Trans Z-order解码"""
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            z = z | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))  # z从最高位提取
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))  # y从中间位提取
            x = x | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))  # x从最低位提取
        return x, y, z

    # === 简化版Hilbert曲线 ===
    def xyz2key_hilbert_simple(self, x, y, z, depth):
        """简化版3D Hilbert曲线编码 - 避免索引问题"""
        key = torch.zeros_like(x)
        for i in range(depth):
            # 提取当前层的坐标位
            x_bit = (x >> i) & 1
            y_bit = (y >> i) & 1
            z_bit = (z >> i) & 1
            
            # 简化的Hilbert变换（避免复杂的查找表）
            hilbert_code = self._simple_hilbert_transform(x_bit, y_bit, z_bit, i)
            key = key | (hilbert_code << (3 * i))
        return key

    def key2xyz_hilbert_simple(self, key, depth):
        """简化版3D Hilbert曲线解码"""
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            hilbert_code = (key >> (3 * i)) & 7  # 提取3位
            x_bit, y_bit, z_bit = self._simple_hilbert_inverse(hilbert_code, i)
            x = x | (x_bit << i)
            y = y | (y_bit << i)
            z = z | (z_bit << i)
        return x, y, z

    def _simple_hilbert_transform(self, x, y, z, level):
        """简化的3D Hilbert变换 - 使用数学运算而非查找表"""
        # 基于Gray码的简化Hilbert变换
        gray_x = x ^ (x >> 1)
        gray_y = y ^ (y >> 1) 
        gray_z = z ^ (z >> 1)
        
        # 根据level进行不同的组合
        if level % 3 == 0:
            return gray_x * 4 + gray_y * 2 + gray_z
        elif level % 3 == 1:
            return gray_z * 4 + gray_x * 2 + gray_y
        else:
            return gray_y * 4 + gray_z * 2 + gray_x

    def _simple_hilbert_inverse(self, hilbert_code, level):
        """简化的3D Hilbert逆变换"""
        # 提取各个位
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
        
        # Gray码逆变换
        x = gray_x ^ (gray_x >> 1)
        y = gray_y ^ (gray_y >> 1)
        z = gray_z ^ (gray_z >> 1)
        
        return x, y, z

    # === Trans Hilbert ===
    def xyz2key_trans_hilbert_simple(self, x, y, z, depth):
        """Trans Hilbert: 坐标转置后应用简化Hilbert"""
        # 坐标转置: (x,y,z) -> (z,x,y)
        return self.xyz2key_hilbert_simple(z, x, y, depth)

    def key2xyz_trans_hilbert_simple(self, key, depth):
        """Trans Hilbert解码"""
        z, x, y = self.key2xyz_hilbert_simple(key, depth)
        return x, y, z  # 转置回来: (z,x,y) -> (x,y,z)


# 全局实例
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


def multi_key2xyz(key: torch.Tensor, depth: int = 16, method: str = 'z_order'):
    """多种序列化方法的统一解码接口"""
    
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