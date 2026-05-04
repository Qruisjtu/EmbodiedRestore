#存储失真图像样例
import distort_lib
import json
if __name__ == "__main__":

    distorter = distort_lib.ImageDistortion("example.png")
    # distorter = distort_lib.ImageDistortion('3.jpg')

    custom_params = {
        'additive_gaussian_noise': {"mean": 0, "std": 25},
        'color_component_noise': {"luminance_std": 10, "color_std": 30},
        'spatially_correlated_noise': {"correlation_strength": 0.8},
        'masked_noise': {"mask_threshold": 128, "noise_std": 50},
        'high_frequency_noise': {"frequency_cutoff": 0.3, "noise_std": 30},
        'impulse_noise': {"salt_prob": 0.01, "pepper_prob": 0.01},
        'quantization_noise': {"levels": 16},
        'gaussian_blur': {'kernel_size': 5, 'sigma': 1.5},
        'image_denoising': {'method': 'bilateral'},
        'jpeg_compression': {'quality': 50},
        'jpeg2000_compression': {'quality': 50},
        'jpeg_transmission_errors': {"error_prob": 0.01},
        'jpeg2000_transmission_errors': {"error_prob": 0.01},
        'non_eccentricity_pattern_noise': {"pattern_strength": 0.1},
        'local_block_distortions': {'block_size': 32, 'distortion_intensity': 0.3},
        'mean_shift': {"shift_value": 50},
        'contrast_change': {'contrast_factor': 1.5},
        'color_saturation_change': {'saturation_factor': 1.5},
        'multiplicative_gaussian_noise': {"mean": 1, "std": 0.1},
        'comfort_noise': {"noise_level": 0.1},
        'lossy_compression_noisy_images': {"noise_std": 20, "compression_quality": 30},
        'color_quantization_dither': {"levels": 8},
        'chromatic_aberrations': {'aberration_strength': 2},
        'sparse_sampling_reconstruction': {'sampling_ratio': 0.7},
        "ringing_artifacts": {"strength": 0.3, "frequency_scale": 5.0},
    }
    strong_params = json.load(open("strong_params.json", "r"))
    distorter.save_imgs(params=strong_params)