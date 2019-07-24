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

#import "PaddleMobileGPU.h"

#import <Foundation/Foundation.h>
#import <paddle_mobile_demo-Swift.h>

@implementation ModelConfig
@end

@interface PaddleMobileGPUResult ()

@property (strong, nonatomic) ResultHolder *resultHolder;

- (void)setOutputResult:(ResultHolder *)resultHolder;

@end

@implementation PaddleMobileGPUResult
- (void)setOutputResult:(ResultHolder *)resultHolder {
    self.resultHolder = resultHolder;
    self.output = resultHolder.result;
    self.outputSize = resultHolder.capacity;
}

-(void)releaseOutput {
    [self.resultHolder releasePointer];
}
@end

@interface PaddleMobileGPU ()
{
    Runner *runner;
}
@end

@implementation PaddleMobileGPU

-(instancetype)initWithCommandQueue:(id<MTLCommandQueue>)queue net:(NetType)netType modelConfig:(ModelConfig *)config {
    self = [super init];
    if (self) {
        Net *net = nil;
        NSError *error = nil;
        if (netType == SuperResolutionNetType) {
            net = [[SuperResolutionNet alloc] initWithDevice:queue.device inParamPointer:config.paramPointer inParamSize:config.paramSize inModelPointer:config.modelPointer inModelSize:config.modelSize error:&error];
        } else if (netType == MobileNetSSDType) {
            net = [[MobileNet_ssd_AR alloc] initWithDevice:queue.device inParamPointer:config.paramPointer inParamSize:config.paramSize inModelPointer:config.paramPointer inModelSize:config.modelSize error:&error];
        }
        if (!error && net) {
            runner = [[Runner alloc] initInNet:net commandQueue:queue error:&error];
        }
    }
    return self;
}

-(BOOL)load {
    return [runner load];
}

-(void)predict:(id<MTLTexture>)texture withCompletion:(void (^)(BOOL, NSArray<NSArray <NSNumber *>*> *))completion {
    
    [runner predictWithTexture:texture completion:^(BOOL success, NSArray<ResultHolder *> * _Nullable resultArr) {
        NSMutableArray<NSMutableArray <NSNumber *>*> *ocResultArray = [NSMutableArray arrayWithCapacity:resultArr.count];
        for (int i = 0; i < resultArr.count; ++i) {
            ResultHolder *resultHolder = resultArr[i];
            NSMutableArray <NSNumber *>*res = [NSMutableArray arrayWithCapacity:resultHolder.capacity];
            for (int j = 0; j < resultHolder.capacity; ++j) {
                [res addObject:[NSNumber numberWithFloat:resultHolder.result[i]]];
            }
            [ocResultArray addObject:res];
            [resultHolder releasePointer];
        }
        completion(success, ocResultArray);
    }];
}

-(void)predict:(id<MTLTexture>)texture withResultCompletion:(void (^)(BOOL, NSArray <PaddleMobileGPUResult *> *))completion {
    [runner predictWithTexture:texture completion:^(BOOL success, NSArray<ResultHolder *> * _Nullable resultArr) {
        NSMutableArray <PaddleMobileGPUResult *> *ocResultArr = [NSMutableArray arrayWithCapacity:resultArr.count];
        for (int i = 0; i < resultArr.count; ++i) {
            ResultHolder *result = resultArr[i];
            PaddleMobileGPUResult *gpuResult = [[PaddleMobileGPUResult alloc] init];
            gpuResult.dim = result.dim;
            [gpuResult setOutputResult:result];
            [ocResultArr addObject:gpuResult];
        }
        completion(success, ocResultArr);
    }];
}

-(void)clear {
    [runner clear];
}

@end
