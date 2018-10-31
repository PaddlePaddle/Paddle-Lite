# iOS开发文档

## CPU

需要: xcode

### 编译

```sh

# 在 paddle-mobile 目录下:
cd tools

sh build.sh ios

# 如果只想编译某个特定模型的 op, 则需执行以下命令
sh build.sh ios googlenet

# 在这个文件夹下, 你可以拿到生成的 .a 库
cd ../build/release/ios/build

```
#### 常见问题:

1. No iOS SDK's found in default search path ...

    这个问题是因为 tools/ios-cmake/ios.toolchain.cmake 找不到你最近使用的 iOS SDK 路径, 所以需要自己进行指定, 
    以我当前的环境为例: 在 tools/ios-cmake/ios.toolchain.cmake 143行前添加我本地的 iOS SDK 路径: set(CMAKE_IOS_SDK_ROOT "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk")

### 集成

```
将上一步生成的:
libpaddle-mobile.a

/src/ios_io/ 下的
PaddleMobileCPU.h
```
拖入工程

#### oc 接口

接口如下:

```
/*
	创建对象
*/
- (instancetype)init;

/*
	load 模型, 开辟内存
*/
- (BOOL)load:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath;

/*
	进行预测, means 和 scale 为训练模型时的预处理参数, 如训练时没有做这些预处理则直接使用 predict
*/
- (NSArray *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim means:(NSArray<NSNumber *> *)means scale:(float)scale;

/*
	进行预测
*/
- (NSArray *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim;

/*
	清理内存
*/
- (void)clear;

```

## GPU

需要: xcode、cocoapods  

```
# 在 paddle-mobile 目录下:
cd metal

pod install

open paddle-mobile.xcworkspace

```
