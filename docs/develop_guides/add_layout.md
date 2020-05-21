# 新增Layout

Paddle-Lite中Place包含了Target、Layout、Precision信息，用来注册和选择模型中的具体Kernel。下面以增加Place中的layout：`ImageDefault`、`ImageFolder`、`ImageNW`为例，讲解如何增加新Layout。

根据在`lite/core/`、`lite/api`目录下以`NHWC`为关键词检索代码，发现需要分别在以下的文件中加入Layout内容：

1. lite/api/paddle_place.h
2. lite/api/paddle_place.cc
3. lite/api/python/pybind/pybind.cc
4. lite/core/op_registry.h
5. lite/core/op_registry.cc

## 1. lite/api/paddle_place.h

在`enum class DataLayoutType`中加入对应的Layout，注意已有的Layout不能改变值，增加新Layout递增即可：

```cpp
enum class DataLayoutType : int {
  kUnk = 0,
  kNCHW = 1,
  kNHWC = 3,
  kImageDefault = 4,  // for opencl image2d
  kImageFolder = 5,   // for opencl image2d
  kImageNW = 6,       // for opencl image2d
  kAny = 2,           // any data layout
  NUM = 7,            // number of fields.
};
```

## 2. lite/api/paddle_place.cc

本文件有3处修改，注意在` DataLayoutToStr`函数中加入对应Layout的字符串名，顺序为`lite/api/paddle_place.h`中枚举值的顺序：

```cpp
// 该文件第1处
const std::string& DataLayoutToStr(DataLayoutType layout) {
  static const std::string datalayout2string[] = {
      "unk", "NCHW", "any", "NHWC", "ImageDefault", "ImageFolder", "ImageNW"};
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
                                                  "kImageNW"};
  auto x = static_cast<int>(layout);
  CHECK_LT(x, static_cast<int>(DATALAYOUT(NUM)));
  return datalayout2string[x];
}

// 该文件第3处
std::set<DataLayoutType> ExpandValidLayouts(DataLayoutType layout) {
  static const std::set<DataLayoutType> valid_set({DATALAYOUT(kNCHW),
                                                   DATALAYOUT(kAny),
                                                   DATALAYOUT(kNHWC),
                                                   DATALAYOUT(kImageDefault),
                                                   DATALAYOUT(kImageFolder),
                                                   DATALAYOUT(kImageNW)});
  if (layout == DATALAYOUT(kAny)) {
    return valid_set;
  }
  return std::set<DataLayoutType>({layout});
}
```

## 3. lite/api/python/pybind/pybind.cc

```cpp
  // DataLayoutType
  py::enum_<DataLayoutType>(*m, "DataLayoutType")
      .value("NCHW", DataLayoutType::kNCHW)
      .value("NHWC", DataLayoutType::kNHWC)
      .value("ImageDefault", DataLayoutType::kImageDefault)
      .value("ImageFolder", DataLayoutType::kImageFolder)
      .value("ImageNW", DataLayoutType::kImageNW)
      .value("Any", DataLayoutType::kAny);
```

## 4. lite/core/op_registry.h

找到KernelRegister final中的`using any_kernel_registor_t =`，加入下面修改信息：

```cpp
// 找到KernelRegister final中的`using any_kernel_registor_t =`
// 加入如下内容：
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageNW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageFolder)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageNW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageDefault)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageFolder)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL),
                                      PRECISION(kAny),
                                      DATALAYOUT(kImageNW)> *,  //
```


## 5. lite/core/op_registry.cc

该文件有2处修改：

```cpp
// 该文件第1处
#define CREATE_KERNEL1(target__, precision__)                                \
  switch (layout) {                                                          \
    case DATALAYOUT(kNCHW):                                                  \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kNCHW)>(op_type);                             \
    case DATALAYOUT(kAny):                                                   \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kAny)>(op_type);                              \
    case DATALAYOUT(kNHWC):                                                  \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kNHWC)>(op_type);                             \
    case DATALAYOUT(kImageDefault):                                          \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kImageDefault)>(op_type);                     \
    case DATALAYOUT(kImageFolder):                                           \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kImageFolder)>(op_type);                      \
    case DATALAYOUT(kImageNW):                                               \
      return Create<TARGET(target__),                                        \
                    PRECISION(precision__),                                  \
                    DATALAYOUT(kImageNW)>(op_type);                          \
    default:                                                                 \
      LOG(FATAL) << "unsupported kernel layout " << DataLayoutToStr(layout); \
  }

// 该文件第2处
// 找到文件中的下面的函数
KernelRegistry::KernelRegistry()
    : registries_(static_cast<int>(TARGET(NUM)) *
                  static_cast<int>(PRECISION(NUM)) *
                  static_cast<int>(DATALAYOUT(NUM)))

// 在该函数中加入新增Layout的下面内容
  INIT_FOR(kOpenCL, kFP16, kNCHW);
  INIT_FOR(kOpenCL, kFP16, kNHWC);
  INIT_FOR(kOpenCL, kFP16, kImageDefault);
  INIT_FOR(kOpenCL, kFP16, kImageFolder);
  INIT_FOR(kOpenCL, kFP16, kImageNW);
  INIT_FOR(kOpenCL, kFloat, kImageDefault);
  INIT_FOR(kOpenCL, kFloat, kImageFolder);
  INIT_FOR(kOpenCL, kFloat, kImageNW);
  INIT_FOR(kOpenCL, kAny, kImageDefault);
  INIT_FOR(kOpenCL, kAny, kImageFolder);
  INIT_FOR(kOpenCL, kAny, kImageNW);
```
