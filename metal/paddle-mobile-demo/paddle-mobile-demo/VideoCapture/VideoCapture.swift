
import UIKit
import Metal
import CoreVideo
import AVFoundation

@available(iOS 10.0, *)
@objc public protocol VideoCaptureDelegate: NSObjectProtocol {
    @objc optional func videoCapture(_ capture: VideoCapture, didCaptureSampleBuffer sampleBuffer: CMSampleBuffer, timestamp: CMTime)
    @objc optional func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime)
    @objc optional func videoCapture(_ capture: VideoCapture, didCapturePhoto previewImage: UIImage?)
    @objc optional func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?)
}

/**
 Simple interface to the iPhone's camera.
 */
@available(iOS 10.0, *)
public class VideoCapture: NSObject {
    public var previewLayer: AVCaptureVideoPreviewLayer?
    public weak var delegate: VideoCaptureDelegate?
    public var fps = -1
    private let device: MTLDevice?
    private let videoOrientation: AVCaptureVideoOrientation
    private var textureCache: CVMetalTextureCache?
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private let queue = DispatchQueue(label: "net.machinethink.camera-queue")
    private var lastTimestamp = CMTime()
    private let cameraPosition: AVCaptureDevice.Position
    public init(device: MTLDevice? = nil, orientation: AVCaptureVideoOrientation = .portrait, position: AVCaptureDevice.Position = .back) {
        self.device = device
        self.videoOrientation = orientation
        self.cameraPosition = position
        super.init()
    }
    
    public func setUp(sessionPreset: AVCaptureSession.Preset = .medium,
                      completion: @escaping (Bool) -> Void) {
        queue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }
    
    func fontCamera() -> AVCaptureDevice? {
        let deveices = AVCaptureDevice.DiscoverySession.init(deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video, position: .front).devices
        return deveices.first
        
    }
    
    func setUpCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
        if let inDevice = device{
            guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, inDevice, nil, &textureCache) == kCVReturnSuccess else {
                print("Error: could not create a texture cache")
                return false
            }
        }
        
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset
        
        var oCaptureDevice: AVCaptureDevice?
        switch cameraPosition {
        case .back:
            oCaptureDevice = AVCaptureDevice.default(for: AVMediaType.video)
            break
        case .front:
            oCaptureDevice = fontCamera()
            break
        default:
            break
        }
        
        guard let captureDevice = oCaptureDevice else {
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
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect
        previewLayer.connection?.videoOrientation = self.videoOrientation
        self.previewLayer = previewLayer
        
        let settings: [String : Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]
        
        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        // We want the buffers to be in portrait orientation otherwise they are
        // rotated by 90 degrees. Need to set this _after_ addOutput()!
        videoOutput.connection(with: AVMediaType.video)?.videoOrientation = self.videoOrientation
        
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
    
    /* Captures a single frame of the camera input. */
    public func capturePhoto() {
        let settings = AVCapturePhotoSettings(format: [kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)])
        settings.previewPhotoFormat = [
            kCVPixelBufferPixelFormatTypeKey as String: settings.__availablePreviewPhotoPixelFormatTypes[0],
            kCVPixelBufferWidthKey as String: 480,
            kCVPixelBufferHeightKey as String: 360,
        ]
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    func convertToMTLTexture(sampleBuffer: CMSampleBuffer?) -> MTLTexture? {
        if let textureCache = textureCache, let sampleBuffer = sampleBuffer, let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            var texture: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, imageBuffer, nil, .bgra8Unorm, width, height, 0, &texture)
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


@available(iOS 10.0, *)
extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Because lowering the capture device's FPS looks ugly in the preview,
        // we capture at full speed but only call the delegate at its desired
        // framerate. If `fps` is -1, we run at the full framerate.
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let deltaTime = timestamp - lastTimestamp
        if fps == -1 || deltaTime >= CMTimeMake(1, Int32(fps)) {
            lastTimestamp = timestamp
            self.delegate?.videoCapture?(self, didCaptureSampleBuffer: sampleBuffer, timestamp: timestamp)
            if self.delegate?.responds(to: #selector(VideoCaptureDelegate.videoCapture(_:didCaptureVideoTexture:timestamp:))) ?? false{
                let texture = convertToMTLTexture(sampleBuffer: sampleBuffer)
                delegate?.videoCapture?(self, didCaptureVideoTexture: texture, timestamp: timestamp)
            }
        }
    }
    
    public func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        print("dropped frame")
    }
}

@available(iOS 10.0, *)
extension VideoCapture: AVCapturePhotoCaptureDelegate {
    public func photoOutput(_ captureOutput: AVCapturePhotoOutput,
                            didFinishProcessingPhoto photoSampleBuffer: CMSampleBuffer?,
                            previewPhoto previewPhotoSampleBuffer: CMSampleBuffer?,
                            resolvedSettings: AVCaptureResolvedPhotoSettings,
                            bracketSettings: AVCaptureBracketedStillImageSettings?,
                            error: Error?) {
        var imageTexture: MTLTexture?
        var previewImage: UIImage?
        if error == nil {
            if self.delegate?.responds(to: #selector(VideoCaptureDelegate.videoCapture(_:didCapturePhotoTexture:))) ?? false{
                imageTexture = convertToMTLTexture(sampleBuffer: photoSampleBuffer)
                self.delegate?.videoCapture?(self, didCapturePhotoTexture: imageTexture)
            }
            
            if self.delegate?.responds(to: #selector(VideoCaptureDelegate.videoCapture(_:didCapturePhoto:))) ?? false{
                previewImage = convertToUIImage(sampleBuffer: previewPhotoSampleBuffer)
                self.delegate?.videoCapture?(self, didCapturePhoto: previewImage)
            }
        }
    }
}
