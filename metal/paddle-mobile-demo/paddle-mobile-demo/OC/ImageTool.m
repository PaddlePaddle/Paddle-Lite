//
//  ImageTool.m
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2019/3/12.
//  Copyright © 2019 orange. All rights reserved.
//

#import "ImageTool.h"

@implementation ImageTool

+ (CVPixelBufferRef)imageToRGBPixelBuffer:(UIImage *)image {
    CVPixelBufferRef pxbuffer = NULL;
#if defined(__arm__) || defined(__arm64__)
    CGSize frameSize = CGSizeMake(CGImageGetWidth(image.CGImage),CGImageGetHeight(image.CGImage));
    //Metal渲染纹理需要IOSurface属性
    NSDictionary *options =
    [NSDictionary dictionaryWithObjectsAndKeys:
     [NSNumber numberWithBool:YES], kCVPixelBufferCGImageCompatibilityKey,
     [NSNumber numberWithBool:YES], kCVPixelBufferCGBitmapContextCompatibilityKey,
     [NSNumber numberWithBool:YES], kCVPixelBufferIOSurfaceOpenGLESTextureCompatibilityKey,
     [NSNumber numberWithBool:YES], kCVPixelBufferIOSurfaceCoreAnimationCompatibilityKey,nil];
    CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width, frameSize.height,kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)options, &pxbuffer);
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, frameSize.width, frameSize.height,8, CVPixelBufferGetBytesPerRow(pxbuffer),rgbColorSpace,(CGBitmapInfo)kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image.CGImage),CGImageGetHeight(image.CGImage)), image.CGImage);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
#endif
    
    return pxbuffer;
}

@end
