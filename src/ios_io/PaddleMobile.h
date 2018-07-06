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

@interface PaddleMobile : NSObject

/*
    创建对象
*/
- (instancetype)init;

/*
    load 模型, 开辟内存
*/
- (BOOL)load:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath;

/*
  加载散开形式的模型, 需传入模型的目录
*/
- (BOOL)load:(NSString *)modelAndWeightPath;

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

@end
