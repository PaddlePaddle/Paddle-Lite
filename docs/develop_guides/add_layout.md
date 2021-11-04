# 新增 Layout

下面以增加 `kMetalTexture2DArray`、`kMetalTexture2D` 为例，介绍如何在 Paddle Lite 中增加新的 Layout。

> **首先在 paddle_place 文件中注册 Layout 信息，Paddle Lite 中 Place 包含了 Target、Layout、Precision 信息，用来注册和选择模型中的具体 Kernel。**


## 1. lite/api/paddle_place.h

在 `enum class DataLayoutType` 中加入新的 Layout，注意已有的 Layout 不能改变值，增加新 Layout 递增取值即可：

```cpp
enum class DataLayoutType : int {
  kUnk = 0,
  kNCHW = 1,
  kNHWC = 3,
  kImageDefault = 4,  // for opencl image2d
  kImageFolder = 5,   // for opencl image2d
  kImageNW = 6,       // for opencl image2d
  kAny = 2,           // any data layout
  kMetalTexture2DArray = 7,
  kMetalTexture2D = 8
};
```

## 2. lite/api/paddle_place.cc

本文件有 3 处修改，注意在 `DataLayoutToStr` 函数中加入对应 Layout 的字符串名，顺序为 `lite/api/paddle_place.h` 中枚举值的顺序：

```cpp
// 该文件第1处
const std::string& DataLayoutToStr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"unk",
                                                  "NCHW",
                                                  "any",
                                                  "NHWC",
                                                  "ImageDefault",
                                                  "ImageFolder",
                                                  "ImageNW",
                                                  "MetalTexture2DArray",
                                                  "MetalTexture2D"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

// 该文件第2处
const std::string& DataLayoutRepr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {"kUnk",
                                                  "kNCHW",
                                                  "kAny",
                                                  "kNHWC",
                                                  "kImageDefault",
                                                  "kImageFolder",
                                                  "kImageNW",
                                                  "kMetalTexture2DArray",
                                                  "kMetalTexture2D"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

// 该文件第3处
std::set<DataLayoutType> ExpandValidLayouts(DataLayoutType layout) {
  static const std::set<DataLayoutType> valid_set(
      {DATALAYOUT(kNCHW),
       DATALAYOUT(kAny),
       DATALAYOUT(kNHWC),
       DATALAYOUT(kImageDefault),
       DATALAYOUT(kImageFolder),
       DATALAYOUT(kImageNW),
       DATALAYOUT(kMetalTexture2DArray),
       DATALAYOUT(kMetalTexture2D)});
  if (layout == DATALAYOUT(kAny)) {
    return valid_set;
  }
  return std::set<DataLayoutType>({layout});
}
```

> **接着，在 opt_base 中给对应的 target_repr 添加新增加的 Layout**

## 3. lite/api/tools/opt_base.cc

```cpp
//metal
if (target_repr == "metal") {
  valid_places_.emplace_back(Place{
      TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray)});
  valid_places_.emplace_back(Place{
      TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray)});
}
```

> **最后，以 relu 算子为例，使用新增加的 Layout**

## 4. lite/kernels/metal/image_op/activation_image_compute.mm

```cpp
//relu
REGISTER_LITE_KERNEL(relu,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ActivationImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
```
