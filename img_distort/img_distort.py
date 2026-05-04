import distort_lib
import json
import argparse
import os
import glob

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='应用图像失真效果')

    parser.add_argument('input_path', type=str, help='输入图像路径或文件夹路径')
    parser.add_argument('-o', '--output_dir', type=str, default='./output_imgs',
                        help='输出目录 (默认: ./output_imgs)')
    parser.add_argument('-p', '--params_file', type=str,
                        help='JSON参数文件路径，包含各失真方法的参数')
    parser.add_argument('--batch', action='store_true',
                        help='批处理模式，处理文件夹中的所有图像')
    parser.add_argument('--ext', type=str, default='jpg,jpeg,png,bmp,tiff',
                        help='批处理时处理的图像扩展名，用逗号分隔 (默认: jpg,jpeg,png,bmp,tiff)')

    return parser.parse_args()


def load_params_from_file(params_file):
    """
    从JSON文件加载参数

    Args:
        params_file: JSON参数文件路径

    Returns:
        dict: 参数字典
    """
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"从文件加载参数: {params_file}")
        return params
    except Exception as e:
        print(f"加载参数文件失败: {e}")
        return None


def create_sample_params_file():
    """
    创建示例参数文件
    """
    sample_params = {
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

    with open('sample_params.json', 'w') as f:
        json.dump(sample_params, f, indent=4)

    print("已创建示例参数文件: sample_params.json")


def get_image_files(folder_path, extensions):
    """
    获取文件夹中指定扩展名的所有图像文件

    Args:
        folder_path: 文件夹路径
        extensions: 扩展名列表

    Returns:
        list: 图像文件路径列表
    """
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*.{ext}")
        image_files.extend(glob.glob(pattern))

    # 添加不区分大小写的匹配
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern))

    return sorted(list(set(image_files)))


def batch_process(input_folder, output_dir, params=None, extensions=None):
    """
    批处理文件夹中的所有图像

    Args:
        input_folder: 输入文件夹路径
        output_dir: 输出目录
        params: 参数字典
        extensions: 图像扩展名列表

    Returns:
        int: 处理的图像数量
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

    image_files = get_image_files(input_folder, extensions)

    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图像文件")
        return 0

    print(f"找到 {len(image_files)} 个图像文件:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")

    processed_count = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理图像 {i}/{len(image_files)}: {os.path.basename(image_path)}")

        try:
            distorter = distort_lib.ImageDistortion(image_path)
            distorter.save_imgs(output_dir=output_dir, params=params)
            processed_count += 1
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")

    print(f"\n批处理完成! 成功处理 {processed_count}/{len(image_files)} 个图像")
    return processed_count


def main():
    """
    主函数 - 处理命令行调用
    """
    args = parse_arguments()
    params = None
    if args.params_file:
        params = load_params_from_file(args.params_file)
        if params is None:
            print("使用默认参数")

    if os.path.isfile(args.input_path):
        print(f"处理单张图像: {args.input_path}")

        try:
            distorter = distort_lib.ImageDistortion(args.input_path)
        except Exception as e:
            print(f"加载图像失败: {e}")
            return

        distorter.save_imgs(output_dir=args.output_dir, params=params)

    elif os.path.isdir(args.input_path):
        print(f"批处理文件夹: {args.input_path}")
        extensions = [ext.strip().lower() for ext in args.ext.split(',')]
        print(f"处理的文件扩展名: {', '.join(extensions)}")

        # 执行批处理
        batch_process(
            input_folder=args.input_path,
            output_dir=args.output_dir,
            params=params,
            extensions=extensions
        )

    else:
        print(f"错误: 输入路径不存在: {args.input_path}")
        return


if __name__ == "__main__":

    import sys

    if len(sys.argv) == 1:
        print("使用方法:")
        print("  python img_distort.py input_path [选项]")
        print("\n选项:")
        print("  -o, --output_dir DIR    输出目录 (默认: ./output_imgs)")
        print("  -p, --params_file FILE  JSON参数文件路径")
        print("  --batch                 批处理模式（自动检测文件夹）")
        print("  --ext EXTENSIONS        批处理时处理的图像扩展名 (默认: jpg,jpeg,png,bmp,tiff)")
        print("  --create_sample         创建示例参数文件")
        print("\n示例:")
        print("  单张图像:")
        print("    python img_distort.py e.png")
        print("    python img_distort.py e.png -o my_output -p my_params.json -v")
        print("  批处理:")
        print("    python img_distort.py images/ --batch")
        print("    python img_distort.py images/ -o batch_output -p my_params.json --ext jpg,png")

        if len(sys.argv) > 1 and sys.argv[1] == '--create_sample':
            create_sample_params_file()
    else:
        if '--create_sample' in sys.argv:
            create_sample_params_file()
        else:
            main()