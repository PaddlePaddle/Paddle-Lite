//
//  ImageTool.h
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2019/3/12.
//  Copyright © 2019 orange. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ImageTool : NSObject

+ (CVPixelBufferRef)imageToRGBPixelBuffer:(UIImage *)image;


@end

NS_ASSUME_NONNULL_END
