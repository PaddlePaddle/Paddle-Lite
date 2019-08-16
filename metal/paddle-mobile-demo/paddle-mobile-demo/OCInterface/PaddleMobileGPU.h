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

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

typedef enum : NSUInteger {
    SuperResolutionNetType,
    MobileNetSSDType
} NetType;

@interface PaddleMobileGPUResult: NSObject

@property (assign, nonatomic) float *output;

@property (assign, nonatomic) int outputSize;

@property (strong, nonatomic) NSArray <NSNumber *>*dim;

-(void)releaseOutput;

@end

@interface ModelConfig: NSObject

/*
 * 预处理需要用到的值 (三个)
 */
@property (strong, nonatomic) NSArray<NSNumber *> *means;
/*
 * 预处理需要用到的 scale 值
 */
@property (assign, nonatomic) float scale;

/*
 * 输出维度信息  [n c h w]
 */
@property (strong, nonatomic) NSArray<NSNumber *> *dims;


/*
 * 模型参数内存地址
 */
@property (assign, nonatomic) void *paramPointer;

/*
 * 模型参数占用内存大小 (kb)
 */
@property (assign, nonatomic) int paramSize;

/*
 * 模型内存地址
 */
@property (assign, nonatomic) void *modelPointer;

/*
 * 模型占用内存大小 (kb)
 */
@property (assign, nonatomic) int modelSize;

@end

@interface PaddleMobileGPU: NSObject

/*
 * 初始化
 */
-(instancetype)initWithCommandQueue:(id<MTLCommandQueue>)queue net:(NetType)netType modelConfig:(ModelConfig *)config;

/*
 * paramPointer 模型参数内存地址
 * paramSize    模型参数占用内存大小 (kb)
 * modelPointer 模型内存地址
 * modelSize    模型占用内存大小 (kb)
 */
-(BOOL)load;

/*
 * texture:     需要进行预测的图像转换的 texture
 * completion:  预测完成回调
 */
-(void)predict:(id<MTLTexture>)texture withCompletion:(void (^)(BOOL, NSArray<NSArray <NSNumber *>*> *))completion;

/*
 * texture:     需要进行预测的图像转换的 texture
 * completion:  预测完成回调
 */
-(void)predict:(id<MTLTexture>)texture withResultCompletion:(void (^)(BOOL, NSArray <PaddleMobileGPUResult *> *))completion;

/*
 * 清理内存
 */
-(void)clear;

@end
