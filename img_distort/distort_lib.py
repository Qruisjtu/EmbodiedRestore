import os
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
from io import BytesIO
from PIL import Image
import time
import json
from skimage.metrics import peak_signal_noise_ratio   #用于计算psnr
import matplotlib.pyplot as plt  #用于可视化
STRONG_PARAMS = json.load(open("img_distort/params_restore.json", "r"))
class ImageDistortion:
    def __init__(self, image):
        if isinstance(image, str):
            self.image = cv2.imread(image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.filename = os.path.basename(image)
            self.name_without_ext = os.path.splitext(self.filename)[0]
        else:
            self.image = image.copy()
            # 为数组图像生成默认文件名
            timestamp = int(time.time())
            self.filename = f"image_{timestamp}.png"
            self.name_without_ext = f"image_{timestamp}"

        self.original_image = self.image.copy()

    def reset(self):
        """重置image为original_image"""
        self.image = self.original_image.copy()
        return self.image

    def additive_gaussian_noise(self, mean=0, std=25):
        """
        1. 加性高斯噪声
        """
        noise = np.random.normal(mean, std, self.image.shape)
        noisy_image = self.image.astype(np.float32) + noise
        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def color_component_noise(self, luminance_std=10, color_std=30):
        """
        2. 色彩分量噪声
        """
        ycrcb = cv2.cvtColor(self.image, cv2.COLOR_RGB2YCrCb).astype(np.float32)

        y_noise = np.random.normal(0, luminance_std, ycrcb[:, :, 0].shape)
        cr_noise = np.random.normal(0, color_std, ycrcb[:, :, 1].shape)
        cb_noise = np.random.normal(0, color_std, ycrcb[:, :, 2].shape)

        ycrcb[:, :, 0] += y_noise
        ycrcb[:, :, 1] += cr_noise
        ycrcb[:, :, 2] += cb_noise

        ycrcb = np.clip(ycrcb, 0, 255)
        self.image = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return self.image

    def spatially_correlated_noise(self, correlation_strength=0.8):
        """
        3. 空间相关噪声
        """
        h, w = self.image.shape[:2]
        white_noise = np.random.normal(0, 25, (h, w))

        kernel_size = int(5 * correlation_strength) + 1
        kernel = cv2.getGaussianKernel(kernel_size, kernel_size)
        kernel_2d = np.outer(kernel, kernel)
        correlated_noise = convolve2d(white_noise, kernel_2d, mode='same')

        correlated_noise = correlated_noise * (25 / np.std(correlated_noise))
        noisy_image = self.image.astype(np.float32) + correlated_noise[:, :, np.newaxis]
        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def masked_noise(self, mask_threshold=128, noise_std=50):
        """
        4. 掩蔽噪声
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        mask = gray > mask_threshold

        noise = np.random.normal(0, noise_std, self.image.shape)

        noisy_image = self.image.astype(np.float32)
        for c in range(3):
            noisy_image[:, :, c] = np.where(mask, noisy_image[:, :, c] + noise[:, :, c], noisy_image[:, :, c])

        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def high_frequency_noise(self, frequency_cutoff=0.3, noise_std=30):
        """
        5. 高频噪声
        """
        h, w = self.image.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        frequency = 10 + int(20 * frequency_cutoff)
        high_freq_noise = np.sin(frequency * xx) * np.cos(frequency * yy)
        high_freq_noise = high_freq_noise * noise_std

        noisy_image = self.image.astype(np.float32) + high_freq_noise[:, :, np.newaxis]
        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def impulse_noise(self, salt_prob=0.01, pepper_prob=0.01):
        """
        6. 脉冲噪声
        """
        noisy_image = self.image.copy()

        salt_mask = np.random.random(self.image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255

        pepper_mask = np.random.random(self.image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0

        self.image = noisy_image
        return self.image

    def quantization_noise(self, levels=16):
        """
        7. 量化噪声
        """
        step = 256 // levels
        quantized_image = (self.image // step) * step
        self.image = quantized_image.astype(np.uint8)
        return self.image

    def gaussian_blur(self, kernel_size=5, sigma=1.5):
        """
        8. 高斯模糊
        """
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)
        return self.image

    def image_denoising(self, method='bilateral', **kwargs):
        """
        9. 图像去噪
        """
        if method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            self.image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)

        elif method == 'nlmeans':
            h = kwargs.get('h', 10)
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, h, h, 7, 21)

        return self.image

    def jpeg_compression(self, quality=50):
        """
        10. JPEG压缩
        """
        pil_image = Image.fromarray(self.image)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        self.image = np.array(compressed_image)
        return self.image

    def jpeg2000_compression(self, quality=50):
        """
        11. JPEG2000压缩
        """
        try:
            pil_image = Image.fromarray(self.image)
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG2000', quality_mode='rates', quality_layers=[quality])
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            self.image = np.array(compressed_image)
        except Exception as e:
            print(f"JPEG2000压缩失败: {e}")
            self.jpeg_compression(quality)

        return self.image

    def jpeg_transmission_errors(self, error_prob=0.01):
        """
        12. JPEG传输错误
        """
        self.jpeg_compression(75)

        h, w = self.image.shape[:2]
        block_size = 8

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if np.random.random() < error_prob:
                    i_end = min(i + block_size, h)
                    j_end = min(j + block_size, w)
                    self.image[i:i_end, j:j_end] = np.random.randint(0, 256,
                                                                     (i_end - i, j_end - j, 3))

        return self.image

    def jpeg2000_transmission_errors(self, error_prob=0.01):
        """
        13. JPEG2000传输错误
        """
        self.jpeg2000_compression(75)

        h, w = self.image.shape[:2]
        num_errors = int(error_prob * h * w)

        for _ in range(num_errors):
            i, j = np.random.randint(0, h), np.random.randint(0, w)
            self.image[i, j] = np.random.randint(0, 256, 3)

        return self.image

    def non_eccentricity_pattern_noise(self, pattern_strength=0.1):
        """
        14. 非偏心模式噪声
        """
        h, w = self.image.shape[:2]

        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y)

        pattern = (np.sin(xx) * np.cos(yy) +
                   np.sin(2 * xx) * np.cos(2 * yy) +
                   np.sin(3 * xx) * np.cos(3 * yy)) / 3

        pattern = pattern * pattern_strength * 255

        noisy_image = self.image.astype(np.float32) + pattern[:, :, np.newaxis]
        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def local_block_distortions(self, block_size=32, distortion_intensity=0.3):
        """
        15. 局部块状失真
        """
        h, w = self.image.shape[:2]

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if np.random.random() < 0.3:
                    i_end = min(i + block_size, h)
                    j_end = min(j + block_size, w)

                    distortion_type = np.random.choice(['blur', 'noise', 'brightness'])

                    if distortion_type == 'blur':
                        block = self.image[i:i_end, j:j_end]
                        self.image[i:i_end, j:j_end] = cv2.GaussianBlur(block, (5, 5), 1)

                    elif distortion_type == 'noise':
                        noise = np.random.normal(0, distortion_intensity * 50,
                                                 (i_end - i, j_end - j, 3))
                        self.image[i:i_end, j:j_end] = np.clip(
                            self.image[i:i_end, j:j_end].astype(np.float32) + noise, 0, 255
                        ).astype(np.uint8)

                    elif distortion_type == 'brightness':
                        adjustment = np.random.uniform(0.5, 1.5)
                        self.image[i:i_end, j:j_end] = np.clip(
                            self.image[i:i_end, j:j_end].astype(np.float32) * adjustment, 0, 255
                        ).astype(np.uint8)

        return self.image

    def mean_shift(self, shift_value=50):
        """
        16. 均值偏移
        """
        shifted_image = self.image.astype(np.float32) + shift_value
        self.image = np.clip(shifted_image, 0, 255).astype(np.uint8)
        return self.image

    def contrast_change(self, contrast_factor=1.5):
        """
        17. 对比度变化
        """
        mean_intensity = np.mean(self.image)
        adjusted_image = (self.image.astype(np.float32) - mean_intensity) * contrast_factor + mean_intensity
        self.image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        return self.image

    def color_saturation_change(self, saturation_factor=1.5):
        """
        18. 色彩饱和度变化
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return self.image

    def multiplicative_gaussian_noise(self, mean=1, std=0.1):
        """
        19. 乘性高斯噪声
        """
        noise = np.random.normal(mean, std, self.image.shape)
        noisy_image = self.image.astype(np.float32) * noise
        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def comfort_noise(self, noise_level=0.1):
        """
        20. 舒适噪声
        """
        noise = np.random.normal(0, noise_level * 25, self.image.shape)
        noisy_image = self.image.astype(np.float32) + noise
        self.image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return self.image

    def lossy_compression_noisy_images(self, noise_std=20, compression_quality=30):
        """
        21. 噪声图像的有损压缩
        """
        self.additive_gaussian_noise(0, noise_std)
        self.jpeg_compression(compression_quality)
        return self.image

    def color_quantization_dither(self, levels=8):
        """
        22. 带抖动的颜色量化
        """
        # 将图像转换为浮点数类型
        image_float = self.image.astype(np.float32)
        h, w, c = image_float.shape

        # 计算量化步长
        step = 255.0 / (levels - 1)

        # 为每个颜色通道单独处理
        for channel in range(c):
            # 获取当前通道
            channel_data = image_float[:, :, channel].copy()

            # 逐行处理，但对每行使用向量化操作
            for i in range(h):
                # 获取当前行
                row = channel_data[i, :].copy()

                # 量化当前行
                quantized_row = np.round(row / step) * step

                # 计算量化误差
                error_row = row - quantized_row

                # 更新当前行
                channel_data[i, :] = quantized_row

                # 扩散误差到下一行（如果不是最后一行）
                if i < h - 1:
                    # 获取下一行
                    next_row = channel_data[i + 1, :].copy()

                    # 向左下扩散 (3/16)
                    if w > 1:  # 确保有足够的列
                        next_row[1:] += error_row[:-1] * (3.0 / 16.0)

                    # 直接向下扩散 (5/16)
                    next_row += error_row * (5.0 / 16.0)

                    # 向右下扩散 (1/16)
                    if w > 1:  # 确保有足够的列
                        next_row[:-1] += error_row[1:] * (1.0 / 16.0)

                    # 更新下一行
                    channel_data[i + 1, :] = next_row

                # 向右扩散到当前行的后续像素 (7/16)
                if w > 1:  # 确保有足够的列
                    channel_data[i, 1:] += error_row[:-1] * (7.0 / 16.0)

            # 更新图像通道
            image_float[:, :, channel] = channel_data

        # 确保像素值在有效范围内
        image_float = np.clip(image_float, 0, 255)

        # 转换回uint8类型
        self.image = image_float.astype(np.uint8)
        return self.image

    def chromatic_aberrations(self, aberration_strength=2):
        """
        23. 色差
        """
        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]

        displacement = aberration_strength

        r_shifted = np.roll(r, displacement, axis=1)
        r_shifted = np.roll(r_shifted, displacement, axis=0)

        b_shifted = np.roll(b, -displacement, axis=1)
        b_shifted = np.roll(b_shifted, -displacement, axis=0)

        aberrated_image = np.stack([r_shifted, g, b_shifted], axis=2)
        self.image = aberrated_image
        return self.image

    def sparse_sampling_reconstruction(self, sampling_ratio=0.7):
        """
        24. 稀疏采样和重建
        """
        h, w, c = self.image.shape

        mask = np.random.random((h, w)) < sampling_ratio

        sampled_image = self.image.copy()
        sampled_image[~mask] = 0

        reconstructed_image = np.zeros_like(self.image)
        for channel in range(c):
            reconstructed_image[:, :, channel] = ndimage.median_filter(
                sampled_image[:, :, channel], size=3
            )

        self.image = reconstructed_image
        return self.image

    def ringing_artifacts(self, strength=0.3, frequency_scale=5.0):
        """
        25. 图像振铃失真
        """
        # 将图像转换为浮点数类型
        image_float = self.image.astype(np.float32)
        h, w, c = image_float.shape

        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian_norm = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
        edge_mask = (laplacian_norm > 0.1).astype(np.float32)
        edge_mask_blur = cv2.GaussianBlur(edge_mask, (0, 0), sigmaX=3.0)

        x_coords = np.linspace(0, 2 * np.pi * frequency_scale, w)
        y_coords = np.linspace(0, 2 * np.pi * frequency_scale, h)
        xx, yy = np.meshgrid(x_coords, y_coords)

        ringing_pattern = (
                np.sin(xx) * np.cos(yy) * 0.5 +
                np.sin(2 * xx) * np.cos(2 * yy) * 0.3 +
                np.sin(4 * xx) * np.cos(4 * yy) * 0.2
        )

        ringing_pattern = ringing_pattern * edge_mask_blur
        ringing_pattern = ringing_pattern * strength * 100

        for channel in range(c):
            channel_data = image_float[:, :, channel]
            channel_data += ringing_pattern
            image_float[:, :, channel] = channel_data

        image_float = np.clip(image_float, 0, 255)
        self.image = image_float.astype(np.uint8)
        return self.image

    def calculate_psnr(self, reference_image=None):
        """
        PSNR计算
        """
        if reference_image is None:
            reference_image = self.original_image

        if self.image.shape != reference_image.shape:
            reference_image = cv2.resize(reference_image,
                                         (self.image.shape[1], self.image.shape[0]))

        return peak_signal_noise_ratio(reference_image, self.image)

    def visualize_distortions(self, distortion_list, figsize=(40, 30)):
        """
        可视化，对比原图和失真图
        """
        n_total = len(distortion_list) + 1
        n_cols = 4
        n_rows = (n_total + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image', fontsize=10)
        axes[0, 0].axis('off')

        for idx, (distortion_name, distortion_params) in enumerate(distortion_list):
            self.reset()
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols
            try:
                distortion_method = getattr(self, distortion_name)
                if distortion_params:
                    distorted_image = distortion_method(**distortion_params)
                else:
                    distorted_image = distortion_method()
                psnr = self.calculate_psnr()

                axes[row, col].imshow(distorted_image)
                axes[row, col].set_title(f'{distortion_name}\nPSNR: {psnr:.2f}dB', fontsize=9)
                axes[row, col].axis('off')
            except Exception as e:
                print(f"应用失真 {distortion_name} 时出错: {e}")
                axes[row, col].text(0.5, 0.5, f'Error:\n{str(e)}',
                                    ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{distortion_name} - Failed', fontsize=9)
                axes[row, col].axis('off')

        total_plots = n_rows * n_cols
        for j in range(len(distortion_list) + 1, total_plots):
            row = j // n_cols
            col = j % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def save_imgs(self, output_dir="./output_imgs", params=None):
        """
        存储图像
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 定义失真方法及其参数
        distortion_methods = [
            ("additive_gaussian_noise",
             params.get("additive_gaussian_noise", {"mean": 0, "std": 25}) if params else {"mean": 0, "std": 25}),

            ("color_component_noise",
             params.get("color_component_noise", {"luminance_std": 10, "color_std": 30}) if params else {
                 "luminance_std": 10, "color_std": 30}),

            ("spatially_correlated_noise",
             params.get("spatially_correlated_noise", {"correlation_strength": 0.8}) if params else {
                 "correlation_strength": 0.8}),

            ("masked_noise",
             params.get("masked_noise", {"mask_threshold": 128, "noise_std": 50}) if params else {"mask_threshold": 128,
                                                                                                  "noise_std": 50}),

            ("high_frequency_noise",
             params.get("high_frequency_noise", {"frequency_cutoff": 0.3, "noise_std": 30}) if params else {
                 "frequency_cutoff": 0.3, "noise_std": 30}),

            ("impulse_noise",
             params.get("impulse_noise", {"salt_prob": 0.01, "pepper_prob": 0.01}) if params else {"salt_prob": 0.01,
                                                                                                   "pepper_prob": 0.01}),

            ("quantization_noise",
             params.get("quantization_noise", {"levels": 16}) if params else {"levels": 16}),

            ("gaussian_blur",
             params.get("gaussian_blur", {"kernel_size": 5, "sigma": 1.5}) if params else {"kernel_size": 5,
                                                                                           "sigma": 1.5}),

            ("image_denoising",
             params.get("image_denoising", {"method": "bilateral"}) if params else {"method": "bilateral"}),

            ("jpeg_compression",
             params.get("jpeg_compression", {"quality": 50}) if params else {"quality": 50}),

            ("jpeg2000_compression",
             params.get("jpeg2000_compression", {"quality": 50}) if params else {"quality": 50}),

            ("jpeg_transmission_errors",
             params.get("jpeg_transmission_errors", {"error_prob": 0.01}) if params else {"error_prob": 0.01}),

            ("jpeg2000_transmission_errors",
             params.get("jpeg2000_transmission_errors", {"error_prob": 0.01}) if params else {"error_prob": 0.01}),

            ("non_eccentricity_pattern_noise",
             params.get("non_eccentricity_pattern_noise", {"pattern_strength": 0.1}) if params else {
                 "pattern_strength": 0.1}),

            ("local_block_distortions",
             params.get("local_block_distortions", {"block_size": 32, "distortion_intensity": 0.3}) if params else {
                 "block_size": 32, "distortion_intensity": 0.3}),

            ("mean_shift",
             params.get("mean_shift", {"shift_value": 50}) if params else {"shift_value": 50}),

            ("contrast_change",
             params.get("contrast_change", {"contrast_factor": 1.5}) if params else {"contrast_factor": 1.5}),

            ("color_saturation_change",
             params.get("color_saturation_change", {"saturation_factor": 1.5}) if params else {
                 "saturation_factor": 1.5}),

            ("multiplicative_gaussian_noise",
             params.get("multiplicative_gaussian_noise", {"mean": 1, "std": 0.1}) if params else {"mean": 1,
                                                                                                  "std": 0.1}),

            ("comfort_noise",
             params.get("comfort_noise", {"noise_level": 0.1}) if params else {"noise_level": 0.1}),

            ("lossy_compression_noisy_images",
             params.get("lossy_compression_noisy_images", {"noise_std": 20, "compression_quality": 30}) if params else {
                 "noise_std": 20, "compression_quality": 30}),

            ("color_quantization_dither",
             params.get("color_quantization_dither", {"levels": 8}) if params else {"levels": 8}),

            ("chromatic_aberrations",
             params.get("chromatic_aberrations", {"aberration_strength": 2}) if params else {"aberration_strength": 2}),

            ("sparse_sampling_reconstruction",
             params.get("sparse_sampling_reconstruction", {"sampling_ratio": 0.7}) if params else {
                 "sampling_ratio": 0.7}),

            ("ringing_artifacts",
             params.get("ringing_artifacts", {"strength":0.3, "frequency_scale":5.0}) if params else {"strength":0.3, "frequency_scale":5.0})
        ]

        # 应用并保存每种失真
        for i, (method_name, method_params) in enumerate(distortion_methods, 1):
            try:
                self.reset()
                distortion_method = getattr(self, method_name)
                if method_params:
                    distortion_method(**method_params)
                else:
                    distortion_method()
                psnr = self.calculate_psnr()
                original_path = os.path.join(output_dir, f"{self.name_without_ext}_01.png")
                cv2.imwrite(original_path, cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))

                output_filename = f"{self.name_without_ext}_{i+1:02d}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

                print(f"保存失真图像 {i:02d}/{len(distortion_methods)}: {output_path} (PSNR: {psnr:.2f}dB)")

            except Exception as e:
                print(f"应用失真 {method_name} 时出错: {e}")

        print(f"\n所有失真图像已保存到目录: {output_dir}")
