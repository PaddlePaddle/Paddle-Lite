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
#import "LoadPointerViewController.h"
#import "paddle-mobile-demo-Bridging-Header.h"

#import <Metal/Metal.h>

@interface LoadPointerViewController ()

@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLTexture> texture;
@property (strong, nonatomic) id<MTLCommandQueue> queue;
@property (strong, nonatomic) PaddleMobileGPU *runner;
@property (strong, nonatomic) ModelConfig *modelConfig;

@end

@implementation LoadPointerViewController

- (void)viewDidLoad {
    [super viewDidLoad];
  
  
  self.device = MTLCreateSystemDefaultDevice();
  
  self.queue = [self.device newCommandQueue];
  
    // Do any additional setup after loading the view.
//  NSString *modelPath = [[NSBundle mainBundle] URLForResource:@"genet_model" withExtension:nil].path;
//  NSString *paramPath = [[NSBundle mainBundle] URLForResource:@"genet_params" withExtension:nil].path;
  
  NSString *modelPath = [[NSBundle mainBundle] URLForResource:@"ar_model" withExtension:nil].path;
  NSString *paramPath = [[NSBundle mainBundle] URLForResource:@"ar_params" withExtension:nil].path;

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
//  _modelConfig.means = @[[NSNumber numberWithFloat:128.0], [NSNumber numberWithFloat:128.0], [NSNumber numberWithFloat:128.0]];
//  _modelConfig.scale = 0.017;
//  _modelConfig.dims = @[[NSNumber numberWithFloat:1], [NSNumber numberWithFloat:128.], [NSNumber numberWithFloat:128.0],[NSNumber numberWithFloat:3.0]];
  _modelConfig.means = @[[NSNumber numberWithFloat:103.94], [NSNumber numberWithFloat:116.78], [NSNumber numberWithFloat:123.68]];
  _modelConfig.scale = 1;
  _modelConfig.dims = @[[NSNumber numberWithFloat:1], [NSNumber numberWithFloat:160.], [NSNumber numberWithFloat:160.0],[NSNumber numberWithFloat:3.0]];
  _modelConfig.modelPointer = buffer;
  _modelConfig.modelSize = (int)fileSize;
  _modelConfig.paramPointer = parmaBuffer;
  _modelConfig.paramSize = (int)paramfileSize;
}
- (IBAction)loaderButtonPressed:(id)sender {
//  _runner = [[PaddleMobileGPU alloc] initWithCommandQueue:self.queue net:GenetType modelConfig:_modelConfig];
  _runner = [[PaddleMobileGPU alloc] initWithCommandQueue:self.queue net:MobileNetSSDType modelConfig:_modelConfig];
  
  [_runner load];
}
- (IBAction)predictButtonPressed:(id)sender {
  [self predict];
}

- (id<MTLTexture>) createTextureFromImage:(UIImage*) image device:(id<MTLDevice>) device
{
  image  =[UIImage imageWithCGImage:[image CGImage]
                              scale:[image scale]
                        orientation: UIImageOrientationLeft];
  
  NSLog(@"orientation and size and stuff %ld %f %f", (long)image.imageOrientation, image.size.width, image.size.height);
  
  CGImageRef imageRef = image.CGImage;
  
  size_t width = self.view.frame.size.width;
  size_t height = self.view.frame.size.height;
  
  size_t bitsPerComponent = CGImageGetBitsPerComponent(imageRef);
  size_t bitsPerPixel = CGImageGetBitsPerPixel(imageRef);
  
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(imageRef);
  
  CGImageAlphaInfo alphaInfo = CGImageGetAlphaInfo(imageRef);
  
  //  NSLog(@"%@ %u", colorSpace, alphaInfo);
  
  CGBitmapInfo bitmapInfo = kCGBitmapByteOrderDefault | alphaInfo;
  //    NSLog(@"bitmap info %u", bitmapInfo);
  
  
  CGContextRef context = CGBitmapContextCreate( NULL, width, height, bitsPerComponent, (bitsPerPixel / 8) * width, colorSpace, bitmapInfo);
  
  if( !context )
  {
    NSLog(@"Failed to load image, probably an unsupported texture type");
    return nil;
  }
  
  CGContextDrawImage( context, CGRectMake( 0, 0, width, height ), image.CGImage);
  
  
  MTLPixelFormat format = MTLPixelFormatRGBA8Unorm;
  
  MTLTextureDescriptor *texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                     width:width
                                                                                    height:height
                                                                                 mipmapped:NO];
  id<MTLTexture> texture = [device newTextureWithDescriptor:texDesc];
  
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
               withBytes:CGBitmapContextGetData(context)
             bytesPerRow:4 * width];
  
  return texture;
}

- (void)predict {
  _texture = [self createTextureFromImage:[UIImage imageNamed:@"hand.jpg"] device:self.device];
  NSTimeInterval startTime = [[NSDate date] timeIntervalSince1970];
  NSInteger max = 428;
  for (int i = 0;i < max; i ++) {
    [_runner predict:_texture withCompletion:^(BOOL success , NSArray<NSNumber *> *result) {
      if (success) {
        if (i == max -1) {
          double time = [[NSDate date] timeIntervalSince1970] - startTime;
          time = (time/max)*1000;
          NSLog(@"gap ==== %fms",time);
        }
//        for (int i = 0; i < result.count; i ++) {
//          NSNumber *number = result[i];
//          NSLog(@"result %d = %f:",i, [number floatValue]);
//        }
      }
    }];
  }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
