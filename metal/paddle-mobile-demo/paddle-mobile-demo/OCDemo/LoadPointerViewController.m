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
#import "paddle_mobile_demo-Swift.h"
#import "LoadPointerViewController.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface LoadPointerViewController ()

@property (weak, nonatomic) IBOutlet UIImageView *imageView;

@property (assign, nonatomic) BOOL loaded;
@property (strong, nonatomic) id<MTLTexture> texture;

@property (strong, nonatomic) PaddleMobileGPU *paddleMobile;
@property (strong, nonatomic) ModelConfig *modelConfig;

@end

@implementation LoadPointerViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.imageView.image = [UIImage imageNamed:@"banana.jpeg"];
    
    NSString *modelPath = [[NSBundle mainBundle] URLForResource:@"super_model" withExtension:nil].path;
    NSString *paramPath = [[NSBundle mainBundle] URLForResource:@"super_params" withExtension:nil].path;
    
    long fileSize;
    FILE *fp;
    fp = fopen([modelPath UTF8String], "rb");
    fseek(fp, 0, SEEK_END);
    fileSize = ftell(fp);
    rewind(fp);
    void *buffer = malloc(fileSize);
    fread(buffer, 1, fileSize, fp);
    fclose(fp);
    
    long paramfileSize;
    FILE *parmaFilePointer;
    parmaFilePointer = fopen([paramPath UTF8String], "rb");
    fseek(parmaFilePointer, 0, SEEK_END);
    paramfileSize = ftell(parmaFilePointer);
    rewind(parmaFilePointer);
    void *parmaBuffer = malloc(paramfileSize);
    fread(parmaBuffer, 1, paramfileSize, parmaFilePointer);
    fclose(parmaFilePointer);
    
    _modelConfig = [[ModelConfig alloc] init];
    _modelConfig.modelPointer = buffer;
    _modelConfig.modelSize = (int)fileSize;
    _modelConfig.paramPointer = parmaBuffer;
    _modelConfig.paramSize = (int)paramfileSize;
}
- (IBAction)loaderButtonPressed:(id)sender {
    self.paddleMobile = [[PaddleMobileGPU alloc] initWithCommandQueue:MetalHelper.shared.queue net:SuperResolutionNetType modelConfig:_modelConfig];
    _loaded = [self.paddleMobile load];
    NSLog(@" load 结果: %@", _loaded ? @"成功" : @"失败");
}
- (IBAction)predictButtonPressed:(id)sender {
    [self predict];
}

- (void)predict {
    UIImage *image = self.imageView.image;
    if (!image) {
        NSLog(@" image is nil");
        return;
    }
    id<MTLTexture> texture = [MetalHelper.shared.textureLoader newTextureWithCGImage:image.CGImage options:nil error:nil];
    _texture = texture;
    if (!_texture) {
        NSLog(@" texture is nil");
        return;
    }
    
    if (!self.loaded) {
        NSLog(@" not load ");
        return;
    }
    
    NSTimeInterval startTime = [[NSDate date] timeIntervalSince1970];
    NSInteger max = 1;
    for (int i = 0;i < max; i ++) {
        [self.paddleMobile predict:_texture withCompletion:^(BOOL success , NSArray<NSNumber *> *result) {
            if (success) {
                if (i == max -1) {
                    double time = [[NSDate date] timeIntervalSince1970] - startTime;
                    time = (time/max)*1000;
                    NSLog(@"gap ==== %fms",time);
                }
            }
        }];
    }
}
- (IBAction)clear:(id)sender {
    [self.paddleMobile clear];
    self.loaded = NO;
}

@end
