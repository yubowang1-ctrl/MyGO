import numpy as np
import tensorflow as tf
import random

class SpectrogramAugmentor:
    def __init__(self, 
                 prob_augment=0.8,      # 每个视图进行增强的概率
                 noise_level=0.05,      # 全局噪声强度
                 mask_ratio_range=(0.05, 0.2),  # mask区域占总长宽的比例范围
                 attenuation_range=(0.1, 0.5)   # 减弱系数范围 (保持 10%-50% 的能量)
                 ):
        self.prob = prob_augment
        self.noise_level = noise_level
        self.mask_r_min, self.mask_r_max = mask_ratio_range
        self.att_min, self.att_max = attenuation_range

    def _generate_colored_noise(self, shape, noise_type='gaussian'):
        """
        生成指定颜色的噪声 (H, W, C)
        由于输入已经是频谱图，H轴代表频率。
        - Gaussian (White): 全频段能量均匀
        - Pink: 能量随频率增加而降低 (1/f) -> 听感平衡
        - Blue: 能量随频率增加而升高 (f) -> 高频刺耳
        """
        H, W, C = shape
        
        # 1. 基础高斯白噪声
        noise = np.random.normal(0, 1, shape).astype(np.float32)
        
        if noise_type == 'gaussian':
            return noise
        
        # 2. 构建频率权重向量 (沿着 H 轴)
        # 避免除以 0，从 1 开始
        f = np.linspace(1, H, H).reshape(H, 1, 1).astype(np.float32) 
        
        if noise_type == 'pink':
            # 能量与 1/f 成正比，幅度与 1/sqrt(f) 成正比
            scale = 1.0 / (np.sqrt(f) + 1e-5)
            # 归一化scale以便维持原本的噪声幅度水平
            scale = scale / scale.mean()
            noise = noise * scale
            
        elif noise_type == 'blue':
            # 能量与 f 成正比，幅度与 sqrt(f) 成正比
            scale = np.sqrt(f)
            scale = scale / scale.mean()
            noise = noise * scale
            
        return noise

    def _get_random_roi(self, h, w):
        """随机获取一个矩形区域 (Region of Interest)"""
        # 随机计算区域的宽高
        h_crop = int(h * random.uniform(self.mask_r_min, self.mask_r_max))
        w_crop = int(w * random.uniform(self.mask_r_min, self.mask_r_max))
        
        h_crop = max(1, h_crop)
        w_crop = max(1, w_crop)
        
        # 随机左上角坐标
        h_start = random.randint(0, h - h_crop)
        w_start = random.randint(0, w - w_crop)
        
        return (h_start, h_start + h_crop, w_start, w_start + w_crop), (h_crop, w_crop)

    # === 增强方式 1: 全局噪声 ===
    def apply_global_noise(self, spec):
        noise_type = random.choice(['gaussian', 'pink', 'blue'])
        noise = self._generate_colored_noise(spec.shape, noise_type)
        # 原图 + 噪声系数 * 噪声
        return spec + (self.noise_level * noise)

    # === 增强方式 2: 区域噪声替换 (Mask Replacement) ===
    def apply_region_replacement(self, spec):
        h, w, c = spec.shape
        (h1, h2, w1, w2), (h_crop, w_crop) = self._get_random_roi(h, w)
        
        # 生成对应形状的噪声
        noise_type = random.choice(['gaussian', 'pink', 'blue'])
        noise_patch = self._generate_colored_noise((h_crop, w_crop, c), noise_type)
        
        # 调整噪声的均值和方差，使其与原图该区域的数值范围大致匹配（否则会显得极其突兀）
        # 或者直接使用标准化的高斯噪声，取决于你想让mask多“显眼”
        # 这里选择让噪声强度匹配原图的整体均值，模拟一种“信号丢失变成了白噪声”的感觉
        mean_val = np.mean(spec)
        noise_patch = noise_patch + mean_val 
        
        # 执行替换：spec[x,y] = noise[x,y]
        aug_spec = spec.copy()
        aug_spec[h1:h2, w1:w2, :] = noise_patch
        return aug_spec

    # === 增强方式 3: 区域减弱 (Attenuation) ===
    def apply_region_attenuation(self, spec):
        h, w, c = spec.shape
        (h1, h2, w1, w2), _ = self._get_random_roi(h, w)
        
        # 随机生成减弱系数 (例如 0.3 代表保留 30% 信号)
        coeff = random.uniform(self.att_min, self.att_max)
        
        aug_spec = spec.copy()
        aug_spec[h1:h2, w1:w2, :] *= coeff
        return aug_spec

    def _random_augment(self, spec):
        """随机选择一种增强方式应用"""
        if random.random() > self.prob:
            return spec # 不增强，返回原图

        choice = random.choice(['global_noise', 'replace', 'attenuate'])
        
        if choice == 'global_noise':
            return self.apply_global_noise(spec)
        elif choice == 'replace':
            return self.apply_region_replacement(spec)
        elif choice == 'attenuate':
            return self.apply_region_attenuation(spec)
        return spec

    def _crop_local_view(self, spec, ratio=0.3):
        """生成 Local View (随机切片)"""
        h, w, c = spec.shape
        crop_w = int(w * ratio)
        if w <= crop_w: return spec
        
        start = random.randint(0, w - crop_w)
        return spec[:, start:start+crop_w, :]

    def generate_views(self, origin_np, global_view_num: int, local_view_num: int):
        """
        主入口函数
        
        Args:
            origin_np: 原始频谱图 [H, W, C] (Numpy array)
            global_view_num: 需要生成的全局视图数量 (int)
            local_view_num: 需要生成的局部视图数量 (int)
            
        Returns:
            list[tf.Tensor]: 包含所有视图的列表 (Tensor格式)
        """
        views = []

        # 1. 生成 Global Views
        for _ in range(global_view_num):
            # 复制一份，以免修改原图
            aug_spec = origin_np.copy()
            # 随机增强
            aug_spec = self._random_augment(aug_spec)
            # 转为 Tensor
            views.append(tf.convert_to_tensor(aug_spec, dtype=tf.float32))

        # 2. 生成 Local Views
        for _ in range(local_view_num):
            # 先切片
            local_crop = self._crop_local_view(origin_np, ratio=0.3)
            # 再增强 (在切片后的基础上做增强)
            aug_spec = self._random_augment(local_crop)
            # 转为 Tensor
            views.append(tf.convert_to_tensor(aug_spec, dtype=tf.float32))

        return views
