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

import Metal
import UIKit
import CoreVideo
import Foundation
import AVFoundation

public protocol VideoCaptureDelegate: class {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime)
    func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?)
}

public class VideoCapture: NSObject{
    
    public var previewLayer: AVCaptureVideoPreviewLayer?
    public weak var delegate: VideoCaptureDelegate?
    public var fps = 15
    
    let device: MTLDevice
    var textureCache: CVMetalTextureCache?
    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    let photoOutput = AVCapturePhotoOutput()
    let queue = DispatchQueue(label: "com.codeWorm.videoCaptureQueue")
    
    var lastTimestamp = CMTime()
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    public func setUp(sessionPreset: String = AVCaptureSessionPreset1280x720,
                      completion: @escaping (Bool) -> Void) {
        queue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }
    
    func setUpCamera(sessionPreset: String) -> Bool {
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
            print("Error: could not create a texture cache")
            return false
        }
        
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset
        
        guard let captureDevice = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo) else {
            print("Error: no video devices available")
            return false
        }
        
        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            print("Error: could not create AVCaptureDeviceInput")
            return false
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        if let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession) {
            previewLayer.videoGravity = AVLayerVideoGravityResizeAspect
            previewLayer.connection?.videoOrientation = .portrait
            self.previewLayer = previewLayer
        }
        
        let settings: [String : Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]
        
        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        videoOutput.connection(withMediaType: AVMediaTypeVideo).videoOrientation = .portrait
        
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        }
        
        captureSession.commitConfiguration()
        return true
    }
    
    public func start() {
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
    }
    
    public func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
    
    public func capturePhoto() {
        let settings = AVCapturePhotoSettings(format: [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
            ])
        
        settings.previewPhotoFormat = [
            kCVPixelBufferPixelFormatTypeKey as String: settings.__availablePreviewPhotoPixelFormatTypes[0],
            kCVPixelBufferWidthKey as String: 480,
            kCVPixelBufferHeightKey as String: 360,
        ]
        
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    func convertToMTLTexture(sampleBuffer: CMSampleBuffer?) -> MTLTexture? {
        if let textureCache = textureCache,
            let sampleBuffer = sampleBuffer,
            let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            
            var texture: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
                                                      imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)
            if let texture = texture {
                return CVMetalTextureGetTexture(texture)
            }
        }
        return nil
    }
    
    func convertToUIImage(sampleBuffer: CMSampleBuffer?) -> UIImage? {
        if let sampleBuffer = sampleBuffer,
            let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            let rect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
            
            let ciImage = CIImage(cvPixelBuffer: imageBuffer)
            let ciContext = CIContext(options: nil)
            if let cgImage = ciContext.createCGImage(ciImage, from: rect) {
                return UIImage(cgImage: cgImage)
            }
        }
        return nil
    }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let deltaTime = timestamp - lastTimestamp
        if deltaTime >= CMTimeMake(1, Int32(fps)) {
            lastTimestamp = timestamp
            let texture = convertToMTLTexture(sampleBuffer: sampleBuffer)
            delegate?.videoCapture(self, didCaptureVideoTexture: texture, timestamp: timestamp)
        }
    }
}

extension VideoCapture: AVCapturePhotoCaptureDelegate {
    public func capture(_ captureOutput: AVCapturePhotoOutput,
                        didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?,
                        previewPhotoSampleBuffer: CMSampleBuffer?,
                        resolvedSettings: AVCaptureResolvedPhotoSettings,
                        bracketSettings: AVCaptureBracketedStillImageSettings?,
                        error: Error?) {
        
        var imageTexture: MTLTexture?
        var previewImage: UIImage?
        if error == nil {
            imageTexture = convertToMTLTexture(sampleBuffer: photoSampleBuffer)
            previewImage = convertToUIImage(sampleBuffer: previewPhotoSampleBuffer)
        }
        delegate?.videoCapture(self, didCapturePhotoTexture: imageTexture, previewImage: previewImage)
    }
}
