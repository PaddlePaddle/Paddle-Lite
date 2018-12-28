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

#import "paddle_mobile.h"
#import "PaddleMobileGPU.h"

#import <Foundation/Foundation.h>
#import <paddle_mobile/paddle_mobile-Swift.h>

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
    if (netType == GenetType) {
      net = [[Genet alloc] initWithDevice:queue.device paramPointer:config.paramPointer paramSize:config.paramSize modePointer:config.modelPointer modelSize:config.modelSize];
    } else if (netType == MobileNetSSDType) {
      net = [[MobileNet_ssd_AR alloc] initWithDevice:queue.device paramPointer:config.paramPointer paramSize:config.paramSize modePointer:config.modelPointer modelSize:config.modelSize];
    } else if (netType == MobileNetType) {
      
    }
    runner = [[Runner alloc] initInNet:net commandQueue:queue];
  }
  return self;
}

-(BOOL)load {
  return [runner load];
}

-(void)predict:(id<MTLTexture>)texture withCompletion:(void (^)(BOOL, NSArray<NSNumber *> *))completion {
  
  [runner predictWithTexture:texture completion:^(BOOL success, ResultHolder * _Nullable result) {
    NSMutableArray<NSNumber *> *resultArray = [NSMutableArray arrayWithCapacity:result.capacity];
    for (int i = 0; i < result.capacity; ++i) {
      [resultArray addObject:[NSNumber numberWithFloat:result.result[i]]];
    }
    completion(success, resultArray);
    [result releasePointer];
    
  }];
}

-(void)predict:(id<MTLTexture>)texture withResultCompletion:(void (^)(BOOL, PaddleMobileGPUResult *))completion {
  [runner predictWithTexture:texture completion:^(BOOL success, ResultHolder * _Nullable result) {
    PaddleMobileGPUResult *gpuResult = [[PaddleMobileGPUResult alloc] init];
    [gpuResult setOutputResult:result];
    completion(success, gpuResult);
  }];
}

-(void)clear {
  [runner clear];
}

@end
