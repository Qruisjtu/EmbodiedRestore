#可视化样例
import distort_lib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # distorter = distort_lib.ImageDistortion('example.png')
    distorter = distort_lib.ImageDistortion('3.jpg')

    ## 方法1，列表测试
    distortions_to_test = [
        ('additive_gaussian_noise', {'std': 25}),
        ('additive_gaussian_noise', {'std': 50}),
        ('color_component_noise',{}),
        ('spatially_correlated_noise',{}),
        ('masked_noise',{}),
        ('high_frequency_noise',{}),
        ('impulse_noise', {'salt_prob': 0.02, 'pepper_prob': 0.02}),
        ('quantization_noise', {}),
        ('gaussian_blur', {'kernel_size': 5, 'sigma': 1.5}),
        ('image_denoising', {'method':'bilateral'}),
        ('image_denoising', {'method':'nlmeans'}),
        ('jpeg_compression', {'quality': 30}),
        ('jpeg2000_compression', {'quality': 30}),
        ('jpeg_transmission_errors', {}),
        ('jpeg2000_transmission_errors', {}),
        ('non_eccentricity_pattern_noise', {}),
        ('local_block_distortions', {'block_size':32,'distortion_intensity':0.3}),
        ('local_block_distortions', {'block_size':50,'distortion_intensity':0.4}),
        ('mean_shift', {}),
        ('contrast_change', {'contrast_factor': 2.0}),
        ('color_saturation_change', {'saturation_factor': 0.5}),
        ('multiplicative_gaussian_noise', {}),
        ('comfort_noise', {}),
        ('lossy_compression_noisy_images', {}),
        ('color_quantization_dither', {}),
        ('chromatic_aberrations', {'aberration_strength': 2}),
        ('sparse_sampling_reconstruction', {'sampling_ratio': 0.7}),
        ('sparse_sampling_reconstruction', {'sampling_ratio': 0.5}),
        ("ringing_artifacts", {"strength": 0.3, "frequency_scale": 5.0}),
    ]
    # 可视化对比
    distorter.visualize_distortions(distortions_to_test)

    ## 方法2，单独测试
    print("\n单独测试振铃噪声:")
    distorter.reset()
    noisy_image = distorter.ringing_artifacts()
    psnr = distorter.calculate_psnr()
    print(f"PSNR after Ringing noise: {psnr:.2f}dB")
    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(distorter.original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(noisy_image)
    ax2.set_title(f'With Ringing Noise (PSNR: {psnr:.2f}dB)')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
