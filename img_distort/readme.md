# Readme

## 一、功能介绍
**工具箱实现了25种失真类型，包括了TID2013的24种失真，外加图像振铃失真**
```bash
1 Additive Gaussian noise 加性高斯噪声
2 Color Component noise 色彩分量噪声
3 Spatially correlated noise 空间相关噪声
4 Masked noise 掩蔽噪声
5 High frequency noise 高频噪声
6 Impulse noise 脉冲噪声
7 Quantization noise 量化噪声
8 Gaussian blur 高斯模糊
9 Image denoising 图像去噪
10 JPEG compression JPEG 压缩
11 JPEG2000 compression JPEG2000 压缩
12 JPEG transmission errors JPEG 传输错误
13 JPEG2000 transmission errors JPEG2000 传输错误
14 Non eccentricity pattern noise 非偏心模式噪声
15 Local block-wise distortions 局部块状失真
16 Mean shift 均值偏移
17 Contrast change 对比度变化
18 Change of color saturation 色彩饱和度变化
19 Multiplicative Gaussian noise 乘性高斯噪声
20 Comfort noise 舒适噪声
21 Lossy compression of noisy images 噪声图像的有损压缩
22 Image color quantization with dither 带抖动的颜色量化
23 Chromatic aberrations 色差
24 Sparse sampling and reconstruction 稀疏采样和重建
25 Ringing artifacts 图像振铃失真
```

## 二、快速开始
```bash
  python img_distort.py input_path [选项]
```
### 选项

| 参数 |  注释 |
| ------------ | ------------ |
|-o  --output_dir  DIR |   输出目录 (默认: ./output_imgs)|
|-p  --params_file  FILE | JSON参数文件路径|
|--batch  | 批处理模式（自动检测文件夹）|
|--ext EXTENSIONS   |  批处理时处理的图像扩展名 (默认: jpg,jpeg,png,bmp,tiff)|
|--create_sample   |      创建示例参数文件|

### 示例
####   单张图像
    python img_distort.py e.png
    python img_distort.py e.png -o my_output -p my_params.json -v
####   批处理文件夹
    python img_distort.py images/ --batch
    python img_distort.py images/ -o batch_output -p my_params.json --ext jpg,png

## 三、代码解释
| 代码 |  解释 |
| ------------ | ------------ |
|distort_lib.py| 25种失真代码库，以及PSNR计算、可视化、图像保存|
|img_distort.py| 主函数，有接口可以直接调用|
|example_visual.py|可视化样例|
|example_save.py|输出样例|

