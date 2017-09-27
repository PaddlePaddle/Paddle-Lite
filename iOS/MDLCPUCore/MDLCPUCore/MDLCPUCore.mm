/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/

#include "net.h"
#include "json/json11.h"
#import "MDLCPUCore.h"
#include "math/gemm.h"
#include "base/matrix.h"
#include "loader/loader.h"

#include <mutex>
#include <dirent.h>
#include <iostream>
#import <UIKit/UIKit.h>


@interface MDLCPUCore ()

@property (strong, nonatomic) id<MTLDevice> device;

@property (assign, nonatomic) NSUInteger inThreadNumber;

@end

@implementation MDLCPUCore

using Json = json11::Json;

static mdl::Net *shared_net_instance = nullptr;
static std::mutex shared_mutex;

mdl::Net *get_net_instance(Json &config) {
    if (shared_net_instance == nullptr) {
        shared_net_instance = new mdl::Net(config);
    }
    return shared_net_instance;
}

+ (instancetype)sharedInstance {
    static dispatch_once_t onceToken;
    static id sharedManager = nil;
    dispatch_once(&onceToken, ^{
        sharedManager = [[[self class] alloc] init];
    });
    return sharedManager;
}

- (BOOL)load:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath {
    if (!modelPath || !weighsPath) {
        return NO;
    }
    std::lock_guard<std::mutex> lock(shared_mutex);
    bool success = NO;
    if (mdl::Gemmer::gemmers.size() == 0) {
        for (int i = 0; i < 3; i++) {
            mdl::Gemmer::gemmers.push_back(new mdl::Gemmer());
        }
    }
    std::string *model_path_str = new std::string([modelPath UTF8String]);
    std::string *weights_path_str = new std::string([weighsPath UTF8String]);
    mdl::Loader *loader = mdl::Loader::shared_instance();
    success = loader->load(*model_path_str, *weights_path_str);
    
    delete model_path_str;
    delete weights_path_str;
    
    return success;
}

-(void)preprocess:(const UInt8 *)input output:(float *)output imageWidth:(int)imageWidth imageHeight:(int)imageHeight imageChannels:(int)imageChannels means:(NSArray<NSNumber *> *)means scale:(float)scale{
    int wanted_input_width = 224;
    int wanted_input_height = 224;
    int wanted_input_channels = 3;
    
    for (int c = 0; c < wanted_input_channels; ++c) {
        float *out_channel = output + c * wanted_input_height * wanted_input_width;
        for (int y = 0; y < wanted_input_height; ++y) {
            float *out_row = out_channel + y * wanted_input_width;
            for (int x = 0; x < wanted_input_width; ++x) {
                int in_row = (y * imageHeight) / wanted_input_height;
                int in_col = (x * imageWidth) / wanted_input_width;
                const UInt8 *in_pixel = input + (in_row * imageWidth * imageChannels) + (in_col * imageChannels);
                float *out_pos = out_row + x;
                if (c == 0) {
                    *out_pos = (in_pixel[c] - means[c].floatValue) * scale;
                }else if (c == 1){
                    *out_pos = (in_pixel[c] - means[c].floatValue) * scale;
                }else if (c == 2){
                    *out_pos = (in_pixel[c] - means[c].floatValue) * scale;
                }
            }
        }
    }
}

- (NSArray *)predictImage:(CGImageRef)image means:(NSArray<NSNumber *> *)means scale:(float)scale{
    std::lock_guard<std::mutex> lock(shared_mutex);
    const int sourceRowBytes = CGImageGetBytesPerRow(image);
    const int image_width = CGImageGetWidth(image);
    const int image_height = CGImageGetHeight(image);
    const int image_channels = 4;
    CGDataProviderRef provider = CGImageGetDataProvider(image);
    CFDataRef cfData = CGDataProviderCopyData(provider);
    const UInt8 *input = CFDataGetBytePtr(cfData);
    float *output = (float *)malloc(224*224*3*sizeof(float));
    [self preprocess:input output:output imageWidth:image_width imageHeight:image_height imageChannels:image_channels means:means scale:scale];
    vector<float> cpp_result;
    NSMutableArray *result = NULL;
    long count = 0;
    
    mdl::Loader *loader = mdl::Loader::shared_instance();
    if (!loader->get_loaded()) {
        return nil;
    }
    
    mdl::Net *net = get_net_instance(loader->_model);
    net->set_thread_num([MDLCPUCore sharedInstance].inThreadNumber);
    float *dataPointer = nullptr;
    if (nullptr != output) {
        dataPointer = output;
    }
    cpp_result = net->predict(dataPointer);
    count = cpp_result.size();
    result = [[NSMutableArray alloc] init];
    for (int i = 0; i < count; i++) {
        [result addObject:[NSNumber numberWithFloat:cpp_result[i]]];
    }
    free(output);
    if ([UIDevice currentDevice].systemVersion.doubleValue < 11.0) {
        CFRelease(cfData);
        cfData = NULL;
    }
    return result;
}

- (void)setThreadNumber:(NSUInteger)number{
    [MDLCPUCore sharedInstance].inThreadNumber = number;
}

- (void)clear{
    std::lock_guard<std::mutex> lock(shared_mutex);
    mdl::Loader *loader = mdl::Loader::shared_instance();
    loader->clear();
    if (shared_net_instance) {
        delete shared_net_instance;
        shared_net_instance = nullptr;
    }
}

- (unsigned char *)createRGBABitmapFromImage:(CGImageRef)image{
    CGContextRef context = NULL;
    CGColorSpaceRef colorSpace;
    void *bitmap;
    long bitmapSize;
    long bytesPerRow;
    
    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    
    bytesPerRow   = (width * 4);
    bitmapSize     = (bytesPerRow * height);
    
    bitmap = malloc(bitmapSize);
    if (bitmap == NULL) {
        return NULL;
    }
    
    colorSpace = CGColorSpaceCreateDeviceRGB();
    if (colorSpace == NULL) {
        free(bitmap);
        return NULL;
    }
    
    context = CGBitmapContextCreate (bitmap,
                                     width,
                                     height,
                                     8,
                                     bytesPerRow,
                                     colorSpace,
                                     kCGImageAlphaPremultipliedLast);
    
    CGColorSpaceRelease( colorSpace );
    
    if (context == NULL) {
        free (bitmap);
    }
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    
    return (unsigned char *)bitmap;
}

- (void)dealImage:(CGImageRef)image {
    unsigned char *bitmap = [self createRGBABitmapFromImage:image];
    if (bitmap == NULL) {
        return;
    }
    
}

@end
