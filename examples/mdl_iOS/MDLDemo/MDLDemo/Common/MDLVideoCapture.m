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

#import <Metal/Metal.h>
#import <Photos/Photos.h>
#import <MetalKit/MetalKit.h>

#import "MDLVideoCapture.h"
#import "MDLVideoCapturePreviewView.h"

static const char *kLocalPlayerSessionQueueKey = "session queue";           //队列key
static void * CapturingStillImageContext = &CapturingStillImageContext;
static void * SessionRunningContext = &SessionRunningContext;

typedef NS_ENUM( NSInteger, MDLCaptureSetupResult ) {
    MDLCaptureSetupResultSuccess,
    MDLCaptureSetupResultCameraNotAuthorized,
    MDLCaptureSetupResultSessionConfigurationFailed
};

@interface MDLVideoCapture () <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    CVMetalTextureCacheRef _textureCache;
}
@property (strong, nonatomic) MDLVideoCapturePreviewView *recordingPreviewView;

@property (strong, nonatomic) dispatch_queue_t sessionQueue;

@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) AVCaptureDeviceInput *videoDeviceInput;
@property (strong, nonatomic) AVCaptureVideoDataOutput *videoDataOutput;

@property (assign, nonatomic) CGRect previewFrame;

@property (assign, nonatomic) MDLCaptureSetupResult setupResult;
@property (getter=isSessionRunning, nonatomic) BOOL sessionRunning;


@property (strong, nonatomic) NSDate *lastDate;

@end

@implementation MDLVideoCapture

#pragma mark - 生命周期
- (instancetype)initWithFrame:(CGRect)frame {
    if (self = [super init]) {
        self.previewFrame = frame;
        self.fps = 5;
        self.lastDate = [NSDate date];
    }
    return self;
}

- (UIView *)previewView {
    return self.recordingPreviewView;
}

- (void)registerRecording {
    CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, MTLCreateSystemDefaultDevice(), nil, &_textureCache);
    self.session = [[AVCaptureSession alloc] init];
    
    self.recordingPreviewView = [[MDLVideoCapturePreviewView alloc] initWithFrame:self.previewFrame];
    self.recordingPreviewView.session = self.session;
    self.sessionQueue = dispatch_queue_create( kLocalPlayerSessionQueueKey, DISPATCH_QUEUE_CONCURRENT );
    
    if ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo] == AVAuthorizationStatusNotDetermined) {
        // 挂起队列
        dispatch_suspend(self.sessionQueue);
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^( BOOL granted ) {
            if (!granted) {
                self.setupResult = MDLCaptureSetupResultCameraNotAuthorized;
            }
            dispatch_resume(self.sessionQueue);
        }];
    }
    dispatch_async( self.sessionQueue, ^{
        if (self.setupResult != MDLCaptureSetupResultSuccess) {
            return;
        }
        NSError *error = nil;
        
        AVCaptureDevice *videoDevice = [[self class] deviceWithMediaType:AVMediaTypeVideo preferringPosition:AVCaptureDevicePositionBack];
        AVCaptureDeviceInput *videoDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];
        
        if (!videoDeviceInput) {
            NSLog(@"不能创建输入视频设备: %@", error );
        }
        
        // 开始配置
        [self.session beginConfiguration];
        // 添加视频
        if ( [self.session canAddInput:videoDeviceInput] ) {
            [self.session addInput:videoDeviceInput];
            self.videoDeviceInput = videoDeviceInput;

            
            dispatch_async(dispatch_get_main_queue(), ^{
                UIInterfaceOrientation statusBarOrientation = [UIApplication sharedApplication].statusBarOrientation;
                AVCaptureVideoOrientation initialVideoOrientation = AVCaptureVideoOrientationPortrait;
                if ( statusBarOrientation != UIInterfaceOrientationUnknown ) {
                    initialVideoOrientation = (AVCaptureVideoOrientation)statusBarOrientation;
                }
                
                AVCaptureVideoPreviewLayer *previewLayer = (AVCaptureVideoPreviewLayer *)self.recordingPreviewView.layer;
                previewLayer.connection.videoOrientation = initialVideoOrientation;
            } );
        }
        else {
            NSLog( @"Could not add video device input to the session" );
            self.setupResult = MDLCaptureSetupResultSessionConfigurationFailed;
        }
        
        AVCaptureVideoDataOutput *videoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
        if ([self.session canAddOutput:videoDataOutput]) {
            NSDictionary *setting = @{(__bridge id)kCVPixelBufferPixelFormatTypeKey : [[NSNumber alloc] initWithInt:kCVPixelFormatType_32BGRA]};
            videoDataOutput.videoSettings = setting;
            [videoDataOutput setSampleBufferDelegate:self queue:self.sessionQueue];
            [self.session addOutput:videoDataOutput];
            [videoDataOutput connectionWithMediaType:AVMediaTypeVideo].videoOrientation = AVCaptureVideoOrientationPortrait;
        }
        
        // 提交配置
        [self.session commitConfiguration];
        
        switch (self.setupResult) {
            case MDLCaptureSetupResultSuccess:
            {
                [self addObservers];
                self.captureChangeStatusBlock(CaptureStatusRegisterOK, nil);
                break;
            }
            case MDLCaptureSetupResultCameraNotAuthorized:
            {
                self.captureChangeStatusBlock(CaptureStatusError,error);
                break;
            }
            case MDLCaptureSetupResultSessionConfigurationFailed:
            {
                self.captureChangeStatusBlock(CaptureStatusError,error);
                break;
            }
        }
    });
}

- (void)startRecording {
    if (! self.session.isRunning) {
        [self.session startRunning];
        self.sessionRunning = self.session.isRunning;
    }
}


- (void)stopRecording {
    if ([self.session isRunning]) {
        [self.session stopRunning];
        self.sessionRunning = self.session.isRunning;

    }
}

#pragma mark KVO and Notifications
- (void)addObservers {
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(subjectAreaDidChange:) name:AVCaptureDeviceSubjectAreaDidChangeNotification object:self.videoDeviceInput.device];
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(sessionRuntimeError:) name:AVCaptureSessionRuntimeErrorNotification object:self.session];
    
}

- (void)removeObservers {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    
    [self.session removeObserver:self forKeyPath:@"running" context:SessionRunningContext];
}

- (void)subjectAreaDidChange:(NSNotification *)notification {
    CGPoint devicePoint = CGPointMake(0.5, 0.5);
    [self focusWithMode:AVCaptureFocusModeContinuousAutoFocus exposeWithMode:AVCaptureExposureModeContinuousAutoExposure atDevicePoint:devicePoint monitorSubjectAreaChange:NO];
}

- (void)sessionRuntimeError:(NSNotification *)notification {
    NSError *error = notification.userInfo[AVCaptureSessionErrorKey];
    NSLog( @"Capture session runtime error: %@", error );
    if (error.code == AVErrorMediaServicesWereReset) {
        dispatch_async( self.sessionQueue, ^{
            if ( self.isSessionRunning ) {
                [self.session startRunning];
                self.sessionRunning = self.session.isRunning;
            }
            else {
                dispatch_async( dispatch_get_main_queue(), ^{
                } );
            }
        } );
    }
    else {
    }
}

-(id<MTLTexture>)convertTexture:(CMSampleBufferRef)sampleBuffer{
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    CVMetalTextureRef texture;
    CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, _textureCache, imageBuffer, nil, MTLPixelFormatBGRA8Unorm, width, height, 0, &texture);
    id<MTLTexture> reTexture = CVMetalTextureGetTexture(texture);
    CVBufferRelease(texture);
    return reTexture;
}

#pragma - mark AVCaptureVideoDataOutputSampleBufferDelegate
-(void)captureOutput:(AVCaptureOutput *)output didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection{
   
}

-(void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection{
    NSDate *currentDate = [NSDate date];
    NSTimeInterval timeELe = [currentDate timeIntervalSinceDate:self.lastDate];
    if (timeELe >= 1.0/self.fps) {
        if ([self.delegate respondsToSelector:@selector(captureTexture:)]) {
            [self.delegate captureTexture:[self convertTexture:sampleBuffer]];
        }
        self.lastDate = currentDate;
    }else{
     
    }
}



#pragma mark - Configuration 设备配置
- (void)focusWithMode:(AVCaptureFocusMode)focusMode exposeWithMode:(AVCaptureExposureMode)exposureMode atDevicePoint:(CGPoint)point monitorSubjectAreaChange:(BOOL)monitorSubjectAreaChange {
    dispatch_async( self.sessionQueue, ^{
        AVCaptureDevice *device = self.videoDeviceInput.device;
        NSError *error = nil;
        if ( [device lockForConfiguration:&error] ) {
            if ( device.isFocusPointOfInterestSupported && [device isFocusModeSupported:focusMode] ) {
                device.focusPointOfInterest = point;
                device.focusMode = focusMode;
            }
            
            if ( device.isExposurePointOfInterestSupported && [device isExposureModeSupported:exposureMode] ) {
                device.exposurePointOfInterest = point;
                device.exposureMode = exposureMode;
            }
            
            device.subjectAreaChangeMonitoringEnabled = monitorSubjectAreaChange;
            [device unlockForConfiguration];
        }
        else {
            NSLog( @"Could not lock device for configuration: %@", error );
        }
    } );
}

+ (void)setFlashMode:(AVCaptureFlashMode)flashMode forDevice:(AVCaptureDevice *)device {
    if ( device.hasFlash && [device isFlashModeSupported:flashMode] ) {
        NSError *error = nil;
        if ( [device lockForConfiguration:&error] ) {
            device.flashMode = flashMode;
            [device unlockForConfiguration];
        }
        else {
            NSLog( @"Could not lock device for configuration: %@", error );
        }
    }
}

+ (AVCaptureDevice *)deviceWithMediaType:(NSString *)mediaType preferringPosition:(AVCaptureDevicePosition)position {
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:mediaType];
    AVCaptureDevice *captureDevice = devices.firstObject;
    
    for (AVCaptureDevice *device in devices) {
        if (device.position == position) {
            captureDevice = device;
            break;
        }
    }
    
    return captureDevice;
}

@end
