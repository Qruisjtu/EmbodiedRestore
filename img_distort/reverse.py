import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from distort_lib import ImageDistortion
# ==============================================================================
# 1. 参数搜索网格配置区 (非常方便调整上下限和步长)
# 说明: 
# - 列表里只有一个值的，作为固定参数。
# - 列表里有多个值的，作为搜索参数。
# - 1个变动参数生成 1D 图片行；2个变动参数生成 2D 图片矩阵。
# ==============================================================================
PARAM_GRIDS = {
    "additive_gaussian_noise": {
        "mean": [0], 
        "std": [80,90,100]  # 1D搜索：分细一些
    },
    "color_component_noise": {
        "luminance_std": [20, 30, 40, ], # 2D Matrix Y轴
        "color_std": [60,70,80,90,100]      # 2D Matrix X轴
    },
    "spatially_correlated_noise": {
        "correlation_strength": [  0.8, 1.0, 1.2, 1.5]
    },
    "masked_noise": {
        "mask_threshold": [16, 32, 64, 128], # Y轴
        "noise_std": [30, 60, 100, 120, 150] # X轴
    },
    "high_frequency_noise": {
        "frequency_cutoff": [0.3, 0.6, 0.9],
        "noise_std": [30, 60, 100, 150]
    },
    "impulse_noise": {
        "salt_prob": [0.01, 0.05, 0.1, 0.2],
        "pepper_prob": [0.01, 0.05, 0.1, 0.2]
    },
    "quantization_noise": {
        "levels": [2, 4, 8, 16, 32, 64]
    },
    "gaussian_blur": {
        "kernel_size": [ 11,13,15,17,19,21],      # Y轴 (必须是奇数)
        "sigma": [10.0,12.0, 15.0,17.0] # X轴
    },
    "image_denoising": {
        "method": ["none", "bilateral", "nlmeans"]
    },
    "jpeg_compression": {
        "quality": [2, 5, 10, 20, 30, 50, 75]
    },
    "jpeg2000_compression": {
        "quality": [2, 5, 10, 20, 30, 50, 75]
    },
    "jpeg_transmission_errors": {
        "error_prob": [0.01, 0.05, 0.1,0.15, 0.2, 0.3]
    },
    "jpeg2000_transmission_errors": {
        "error_prob": [0.01, 0.05, 0.1,0.15, 0.2, 0.3]
    },
    "non_eccentricity_pattern_noise": {
        "pattern_strength": [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    },
    "local_block_distortions": {
        "block_size": [ 16, 32],
        "distortion_intensity": [ 3.0,3.5,4.0,4.5,5.0]
    },
    "mean_shift": {
        "shift_value": [30, 60, 90, 120, 150, 180]
    },
    "contrast_change": {
        "contrast_factor": [ 1.5,2.0,2.5,3.0]
    },
    "color_saturation_change": {
        "saturation_factor": [2.0,2.5,3.0,3.5]
    },
    "multiplicative_gaussian_noise": {
        "mean": [1],
        "std": [0.1, 0.3, 0.5, 0.8, 1.2]
    },
    "comfort_noise": {
        "noise_level": [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    },
    "lossy_compression_noisy_images": {
        "noise_std": [ 50, 60,70,80,90,100],
        "compression_quality": [5, 10, 20, 50]
    },
    "color_quantization_dither": {
        "levels": [2, 4, 8, 16, 32]
    },
    "chromatic_aberrations": {
        "aberration_strength": [5,6,7,8,9,10]
    },
    "sparse_sampling_reconstruction": {
        "sampling_ratio": [ 0.5, 0.55,0.6,0.65,0.7]
    },
    "ringing_artifacts": {
        "strength": [0.8, 1.0,1.2],
        "frequency_scale": [10,12,14,16,18,20]
    }
}


# ==============================================================================
# 3. 可视化生成器逻辑
# ==============================================================================
class VisualGridSearch:
    def __init__(self, orig_img_path, output_dir="debug"):
        self.orig_img_path = orig_img_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 预先载入原图
        self.orig_img = cv2.cvtColor(cv2.imread(orig_img_path), cv2.COLOR_BGR2RGB)
        self.distorter = ImageDistortion(self.orig_img)

    def format_params_title(self, params_dict):
        """将参数字典格式化为换行的字符串，适合作为图片标题"""
        return "\n".join([f"{k}: {v}" for k, v in params_dict.items()])

    def generate_debug_images(self):
        print(f"开始生成可视化网格搜索图，将保存至 ./{self.output_dir} 目录...\n")
        
        for method_name, param_grid in PARAM_GRIDS.items():
            print(f"正在处理: {method_name}")
            
            if not hasattr(self.distorter, method_name):
                print(f"  [跳过] ImageDistortion 类中没有找到方法: {method_name}")
                continue

            # 区分变动参数和固定参数
            var_keys = [k for k, v in param_grid.items() if len(v) > 1]
            fixed_params = {k: v[0] for k, v in param_grid.items() if len(v) == 1}

            # ---------------- Case 1: 1D 搜索 (1个变动参数) ----------------
            if len(var_keys) <= 1:
                var_key = var_keys[0] if len(var_keys) == 1 else None
                var_values = param_grid[var_key] if var_key else [None]
                
                n_cols = len(var_values) + 1  # 加1是为了展示原图
                fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 5))
                if n_cols == 1: axes = [axes] # 防止只有一个图时报错
                
                # 画原图
                axes[0].imshow(self.orig_img)
                axes[0].set_title("Original Image", fontsize=12, fontweight='bold', color='blue')
                axes[0].axis('off')

                # 画不同参数的失真图
                for idx, val in enumerate(var_values):
                    current_params = fixed_params.copy()
                    if var_key:
                        current_params[var_key] = val
                    
                    self.distorter.reset()
                    method = getattr(self.distorter, method_name)
                    distorted_img = method(**current_params)
                    
                    col = idx + 1
                    axes[col].imshow(distorted_img)
                    axes[col].set_title(self.format_params_title(current_params), fontsize=10)
                    axes[col].axis('off')
                
                plt.suptitle(f"Method: {method_name}", fontsize=16, fontweight='bold')
                plt.tight_layout()
                save_path = os.path.join(self.output_dir, f"{method_name}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)

            # ---------------- Case 2: 2D Matrix 搜索 (2个变动参数) ----------------
            elif len(var_keys) == 2:
                row_key = var_keys[0]  # Y轴参数
                col_key = var_keys[1]  # X轴参数
                row_vals = param_grid[row_key]
                col_vals = param_grid[col_key]
                
                n_rows = len(row_vals)
                n_cols = len(col_vals)
                
                # 创建画布：宽=列数*4，高=行数*4
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
                
                for r, r_val in enumerate(row_vals):
                    for c, c_val in enumerate(col_vals):
                        current_params = fixed_params.copy()
                        current_params[row_key] = r_val
                        current_params[col_key] = c_val
                        
                        self.distorter.reset()
                        method = getattr(self.distorter, method_name)
                        distorted_img = method(**current_params)
                        
                        ax = axes[r, c] if n_rows > 1 and n_cols > 1 else axes[max(r, c)]
                        ax.imshow(distorted_img)
                        
                        # 设置坐标轴标签 (仅在边缘显示)
                        title_str = f"{row_key}={r_val}\n{col_key}={c_val}"
                        ax.set_title(title_str, fontsize=10)
                        ax.axis('off')
                
                plt.suptitle(f"Matrix Search: {method_name}", fontsize=18, fontweight='bold', y=1.02)
                plt.tight_layout()
                save_path = os.path.join(self.output_dir, f"{method_name}_matrix.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
            else:
                print(f"  [警告] {method_name} 配置了超过 2 个变动参数，暂时不支持 3D 以上的可视化。")

        print(f"\n✅ 所有比对图生成完毕，请打开 ./{self.output_dir}/ 文件夹进行肉眼比对！")

if __name__ == "__main__":
    # 确保路径指向你提供的第一张干净的原图
    ORIGINAL_IMAGE_PATH = "img_distort/output_imgs/example_01.png" 
    
    # 运行可视化网格搜索
    # (运行前请确保你已经把完整的 ImageDistortion 类粘贴到了上方)
    grid_search = VisualGridSearch(ORIGINAL_IMAGE_PATH)
    grid_search.generate_debug_images()