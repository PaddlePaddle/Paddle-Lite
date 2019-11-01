/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#pragma once

#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>

@interface PaddleMobileCPUResult: NSObject

/**
 @b 输出指针
 */
@property (assign, nonatomic, readonly) float *output;

/**
 @b 输出的 float 数
 * */
@property (assign, nonatomic, readonly) int outputSize;

/**
 @b 维度信息, longlongValue
 */
@property (strong, nonatomic, readonly) NSArray <NSNumber *> *dim;

-(void)releaseOutput;

@end

@interface  PaddleMobileCPUConfig: NSObject

/**
 @b 默认为 1, 多线程时, 建议设置为 2
 */
@property (assign, nonatomic) int threadNum;

/**
 @b 是否开启运行时 infershape
 */
@property  (assign, nonatomic) BOOL loddable;

/**
 @b 是否开启模型 op 融合优化
 */
@property  (assign, nonatomic) BOOL optimize;

/**
 @b 是否预测时初始化内存，用于处理可变输入
 */
@property  (assign, nonatomic) BOOL loadWhenPredict;

@end

@interface PaddleMobileCPU : NSObject

/**
 @b 创建对象

 @param config 配置
 @return paddlemobile CPU 对象
 */
- (instancetype)initWithConfig:(PaddleMobileCPUConfig *)config;

/**
 @b 加载模型

 @param modelPath 模型路径
 @param weighsPath 权重路径
 @return 是否加载成功
 */
- (BOOL)loadModel:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath;

/**
 @b 加载散开形式的模型, 需传入模型的目录

 @param modelAndWeightPath 模型和权重的路径
 @return 是否加载成功
 */
- (BOOL)load:(NSString *)modelAndWeightPath;

/**
 @b 从内存中加载模型

 @param modelLen 模型大小(字节数)
 @param modelBuf 模型在内存中的位置
 @param combinedParamsLen 权重大小(字节数)
 @param combinedParamsBuf 权重在内存中的位置
 @return 是否加载成功
 */
- (BOOL)LoadCombinedMemory:(size_t)modelLen
               andModelBuf:(const uint8_t *)modelBuf
         andModelParamsLen:(size_t)combinedParamsLen
      andCombinedParamsBuf:(const uint8_t *)combinedParamsBuf;

/**
 @b 对图像进行预处理, 需要外部开辟 output 内存, 外部释放 output 内存, 每一个像素经过这样的预处理 (x + means) * scale, 其中 x 为像素值

 @param image 输入的图像
 @param output 预处理后的输出
 @param means 预处理中 means
 @param scale 预处理中的 scale
 @param dim 预处理后的维度
 */
-(void)preprocess:(CGImageRef)image
           output:(float *)output
            means:(NSArray<NSNumber *> *)means
        scale:(float)scale
        dim:(NSArray<NSNumber *> *)dim;

/**
 进行预测

 @param input 输入
 @param dim 输入维度
 @return 输出结果
 */
- (PaddleMobileCPUResult *)predictInput:(float *)input
                                    dim:(NSArray<NSNumber *> *)dim;

/**
 @b 进行预测, means 和 scale 为训练模型时的预处理参数, 如训练时没有做这些预处理则直接使用 predict, 每一个像素经过这样的预处理 (x + means) * scale, 其中 x 为像素值

 @param image 输入图像
 @param dim 输入维度
 @param means 预处理中 means
 @param scale 预处理中 scale
 @return 预测结果
 */
- (PaddleMobileCPUResult *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim means:(NSArray<NSNumber *> *)means scale:(float)scale;

/**
 @b 进行预测, means stds和 scale 为训练模型时的预处理参数, 如训练时没有做这些预处理则直接使用 predict, 每一个像素经过这样的预处理 (x + means) * scale, 其中 x 为像素值

 @param image 输入图像
 @param dim 输入维度
 @param means 预处理中 means
 @param stds 预处理中 stds
 @param scale 预处理中 scale
 @return 预测结果
 */
- (PaddleMobileCPUResult *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim means:(NSArray<NSNumber *> *)means stds:(NSArray<NSNumber *> *)stds scale:(float)scale;

/**
 @b 进行预测, 预处理 means 值为 0, scale 值为 1

 @param image 输入图像
 @param dim 输入维度
 @return 预测结果
 */
- (PaddleMobileCPUResult *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim;


/**
 @b 取出模型描述中 key 为 "fetch" 对应的输出

 @return 预测结果
 */
- (PaddleMobileCPUResult *)fetchOutput;

/**
 @b 当输出为多个时, 可用此函数取出对应的输出

 @param key 模型中输出的key
 @return 预测结果
 */
- (PaddleMobileCPUResult *)fetchOutputWithKey:(NSString *)key;

/**
 @b 清理内存
 */
- (void)clear;

@end
