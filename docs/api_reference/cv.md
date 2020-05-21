# CV图像预处理API

请把编译脚本`Paddle-Lite/lite/too/build.sh`中`BUILD_CV`变量设置为`ON`， 其他编译参数设置请参考[源码编译](../user_guides/source_compile)， 以确保 Lite 可以正确编译。这样`CV`图像的加速库就会编译进去，且会生成`paddle_image_preprocess.h`的API文件

- 硬件平台： `ARM`
- 操作系统：`MAC` 和 `LINUX`

## CV 图像预处理功能

Lite 支持不同颜色空间的图像相互转换 `Convert` 、缩放 `Resize` 、翻转 `Flip`、旋转 `Rotate` 和图像数据转换为 `Tensor` 存储`ImageToTensor` 功能，下文将详细介绍每个功能的API接口。

### CV 枚举变量和结构体变量

- 颜色空间
```cpp
enum ImageFormat {
  RGBA = 0,
  BGRA,
  RGB,
  BGR,
  GRAY,
  NV21 = 11,
  NV12,
};
```
- 翻转参数
```cpp
enum FlipParam {
  X = 0,  // flip along the X axis
  Y,      // flip along the Y axis
  XY      // flip along the XY axis
};
```
- 转换参数
```cpp
typedef struct {
  int ih;                // input height
  int iw;                // input width
  int oh;                // outpu theight
  int ow;                // output width
  FlipParam flip_param;  // flip, support x, y, xy
  float rotate_param;    // rotate, support 90, 180, 270
} TransParam;
```

### ImagePreprocess 类的成员变量

`ImagePreprocess` 类含有以下三个私有成员变量，通过构造函数进行初始化。
```cpp
private:
  ImageFormat srcFormat_; // input image color format
  ImageFormat dstFormat_; // output image color format
  TransParam transParam_; // image transform parameter

// init
ImagePreprocess::ImagePreprocess(ImageFormat srcFormat, ImageFormat dstFormat, TransParam param) {
  this->srcFormat_ = srcFormat;
  this->dstFormat_ = dstFormat;
  this->transParam_ = param;
}
```

### 颜色空间转换 Convert

`Convert` 函数支持颜色空间：GRAY、NV12（NV21）、RGB（BGR）和RGBA（BGRA）

+ 目前支持以下颜色空间的相互转换：
    - GRAY2BGR
    - GRAY2RGB
    - BGR2RGB
    - BGRA2BGR
    - BGRA2RGB
    - RGBA2RGB
    - RGBA2BGR
    - BGRA2RGBA

+ 目前支持以下颜色空间的单向转换：
    - NV12—BGR
    - NV21—BGR
    - NV12—RGB
    - NV21—RGB
    - NV12—BGRA
    - NV21—BGRA
    - NV12—RGBA
    - NV21—RGBA

+ `Convert` 功能的API接口
    ```cpp
    // 方法一
    void ImagePreprocess::imageCovert(const uint8_t* src, uint8_t* dst);
    // 方法二
    void ImagePreprocess::imageCovert(const uint8_t* src,
    uint8_t* dst, ImageFormat srcFormat, ImageFormat dstFormat);
    ```

    + 第一个 `imageCovert` 接口，缺省参数来源于 `ImagePreprocess` 类的成员变量。故在初始化 `ImagePreprocess` 类的对象时，必须要给以下成员变量赋值：
        - param srcFormat：`ImagePreprocess` 类的成员变量`srcFormat_`
        - param dstFormat：`ImagePreprocess` 类的成员变量`dstFormat_`
    
    - 第二个`imageCovert` 接口，可以直接使用

### 缩放 Resize

`Resize` 功能支持颜色空间：GRAY、NV12（NV21）、RGB（BGR）和RGBA（BGRA）
`Resize` 功能目前支持的方法：`bilinear`

+ `Resize` 功能的API接口
    ```cpp
    // 方法一
    void ImagePreprocess::imageResize(const uint8_t* src, uint8_t* dst);
    // 方法二
    void ImagePreprocess::imageResize(const uint8_t* src, uint8_t* dst, ImageFormat srcFormat, ImageFormat srcFormat, int srcw, int srch, int dstw, int dsth);
    ```

    + 第一个`imageResize` 接口，缺省参数来源于`ImagePreprocess` 类的成员变量。故在初始化`ImagePreprocess` 类的对象时，必须要给以下成员变量赋值：
        - param srcFormat：`ImagePreprocess` 类的成员变量`dstFormat_`
        - param srcw：`ImagePreprocess` 类的成员变量`transParam_.iw`
        - param srch：`ImagePreprocess` 类的成员变量`transParam_.ih`
        - param dstw：`ImagePreprocess` 类的成员变量`transParam_.ow`
        - param dsth：`ImagePreprocess` 类的成员变量`transParam_.ow`
    
    - 第二个`imageResize` 接口，可以直接使用

### 旋转 Rotate

`Rotate` 功能支持颜色空间：GRAY、RGB（BGR）和RGBA（BGRA）
`Rotate` 功能目前支持的角度：90、180 和 270

+ `Rotate` 功能的API接口
    ```cpp
    // 方法一
    void ImagePreprocess::imageRotate(const uint8_t* src, uint8_t* dst);
    // 方法二
    void ImagePreprocess::imageRotate(const uint8_t* src, uint8_t* dst, ImageFormat srcFormat, ImageFormat srcFormat, int srcw, int srch, float degree);
    ```

    + 第一个`imageRotate` 接口，缺省参数来源于`ImagePreprocess` 类的成员变量。故在初始化`ImagePreprocess` 类的对象时，必须要给以下成员变量赋值：
        - param srcFormat：`ImagePreprocess` 类的成员变量`dstFormat_`
        - param srcw：`ImagePreprocess` 类的成员变量`transParam_.ow`
        - param srch：`ImagePreprocess` 类的成员变量`transParam_.oh`
        - param degree：`ImagePreprocess` 类的成员变量`transParam_.rotate_param`
    
    - 第二个`imageRotate` 接口，可以直接使用

### 翻转 Flip

`Flip` 功能支持颜色空间：GRAY、RGB（BGR）和RGBA（BGRA）
`Flip` 功能目前支持的功能：沿X轴翻转、沿Y轴翻转和沿XY轴翻转

+ `Flip` 功能的API接口
    ```cpp
    // 方法一
    void ImagePreprocess::imageFlip(const uint8_t* src, uint8_t* dst);
    // 方法二
    void ImagePreprocess::imageFlip(const uint8_t* src, uint8_t* dst, ImageFormat srcFormat, ImageFormat srcFormat, int srcw, int srch, FlipParam flip_param);
    ```

    + 第一个`imageFlip` 接口，缺省参数来源于`ImagePreprocess` 类的成员变量。故在初始化`ImagePreprocess` 类的对象时，必须要给以下成员变量赋值：
        - param srcFormat：`ImagePreprocess` 类的成员变量`dstFormat_`
        - param srcw：`ImagePreprocess` 类的成员变量`transParam_.ow`
        - param srch：`ImagePreprocess` 类的成员变量`transParam_.oh`
        - param flip_param：`ImagePreprocess` 类的成员变量`transParam_.flip_param`
    
    - 第二个`imageFlip` 接口，可以直接使用

### Image2Tensor

`Image2Tensor` 功能支持颜色空间：RGB（BGR）和RGBA（BGRA）
`Image2Tensor` 功能目前支持的Layout：`NCHW`和 `NHWC`
`Image2Tensor` 不仅完成图像转换为`Tensor`数据处理，而且还完成了图像数据的归一化处理

+ `Image2Tensor` 功能的API接口
    ```cpp
    // 方法一
    void ImagePreprocess::image2Tensor(const uint8_t* src, Tensor* dstTensor, LayoutType layout, float* means, float* scales);
    // 方法二
    void ImagePreprocess::image2Tensor(const uint8_t* src, Tensor* dstTensor, ImageFormat srcFormat,  srcw, int srch, LayoutType layout, float* means, float* scales;
    ```

    + 第一个`image2Tensor` 接口，缺省参数来源于`ImagePreprocess` 类的成员变量。故在初始化`ImagePreprocess` 类的对象时，必须要给以下成员变量赋值：
        - param srcFormat：`ImagePreprocess` 类的成员变量`dstFormat_`
        - param srcw：`ImagePreprocess` 类的成员变量`transParam_.ow`
        - param srch：`ImagePreprocess` 类的成员变量`transParam_.oh`
    
    - 第二个`image2Tensor` 接口，可以直接使用



## CV 图像预处理 Demo 示例

例子：输入 `1920x1080` 大小的 `NV12` 图像src，输出 `960x540` 大小 `RGB` 格式的图像dst；然后，完成 `90` 度旋转和沿 `X` 轴翻转功能；最后，用 `NHWC` 格式存储在Tensor里。

定义 `ImagePreprocess` 类的对象，初始化成员变量

```cpp
// init
srcFormat = ImageFormat::NV12;
dstFormat = ImageFormat::RGB;
srch = 1920;
srcw = 1080;
dsth = 960;
dstw = 540;
flip_param = FlipParam::X;
degree = 90;
layout = LayoutType::NHWC
// 方法一: 
TransParam tparam;
tparam.ih = srch;
tparam.iw = srcw;
tparam.oh = dsth;
tparam.ow = dstw;
tparam.flip_param = flip_param;
tparam.rotate_param = degree;
ImagePreprocess image_preprocess(srcFormat, dstFormat, tparam);
// 方法二: 
ImagePreprocess image_preprocess();
```

### imageConvert Demo

```cpp
// 方法一: 
image_preprocess.imageCovert(src, lite_dst);
// 方法二: 
image_preprocess.imageCovert(src, lite_dst, (ImageFormat)srcFormat, (ImageFormat)dstFormat);
```

### imageResize Demo

```cpp
// 方法一: 
image_preprocess.imageResize(lite_dst, resize_tmp);
// 方法二: 
image_preprocess.imageResize(lite_dst,resize_tmp, (ImageFormat)dstFormat, srcw,
srch, dstw, dsth);
```

### imageRotate Demo

```cpp
// 方法一: 
image_preprocess.imageRotate(resize_tmp, tv_out_ratote);
// 方法二: 
image_preprocess.imageRotate(resize_tmp,tv_out_ratote, (ImageFormat)dstFormat, dstw, dsth, degree);
```

### imageFlip Demo

```cpp
// 方法一: 
image_preprocess.imageFlip(tv_out_ratote, tv_out_flip);
// 方法二: 
image_preprocess.imageFlip(tv_out_ratote, tv_out_flip, (ImageFormat)dstFormat， dstw, dsth, flip_param);
```

### image2Tensor Demo

```cpp
// 方法一: 
image_preprocess.image2Tensor(tv_out_flip, &dst_tensor, layout, means, scales);
// 方法二: 
image_preprocess.image2Tensor(tv_out_flip, &dst_tensor,(ImageFormat)dstFormat, dstw, dsth, layout, means, scales);
```
