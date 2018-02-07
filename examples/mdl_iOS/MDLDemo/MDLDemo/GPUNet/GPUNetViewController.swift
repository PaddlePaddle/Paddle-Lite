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

import MDL
import UIKit
import Metal
import MetalKit
import CoreMedia
import MetalPerformanceShaders

enum GPUModelType {
    case squeezeNet, mobileNet
}
var gpuModelType: GPUModelType = .mobileNet

@available(iOS 10.0, *)
class GPUNetViewController: UIViewController, CaptureTextureDelegate {
    
    @IBOutlet weak var videoView: UIView!
    @IBOutlet weak var resultLabel: UILabel!
    let labels = PreWords(fileName: "pre_words")
    let device = MTLCreateSystemDefaultDevice()!
    var textureLoader: MTKTextureLoader!
    var commandQueue: MTLCommandQueue!
    var net: MDLGPUNet?
    var isFirstIn = true
    var videoCapture: MDLVideoCapture!

    override func viewDidLoad() {
        super.viewDidLoad()
        commandQueue = device.makeCommandQueue()
        textureLoader = MTKTextureLoader(device: device)
        initCapture {
            self.initializeNet {
                self.videoCapture.startRecording()
            }
        }
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        videoCapture.stopRecording()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        if !isFirstIn {
            videoCapture.startRecording()
        }
        isFirstIn = false
    }
    
    func initCapture(success: @escaping () -> Void) {
        videoCapture = MDLVideoCapture.init(frame: self.videoView.bounds)
        videoCapture.captureChangeStatusBlock = {(status) in
            success()
        }
        videoCapture.fps = 15
        videoCapture.delegate = self
        videoCapture.registerRecording()
        if let preView = videoCapture.previewView(){
            self.videoView.insertSubview(preView, at: 0)
            preView.frame = self.videoView.bounds
        }
    }
    
    
    func initializeNet(completion: @escaping () -> Void) {
        guard MPSSupportsMTLDevice(device) else {
//            fatalError("设备不支持")
            let maskView = UIView.init(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.size.width, height: UIScreen.main.bounds.size.height))
            maskView.backgroundColor = UIColor.white
            let maskLabelView = UILabel()
            maskLabelView.text = "Your device does not support metal"
            maskView.addSubview(maskLabelView)
            maskLabelView.textColor = UIColor.black
            maskLabelView.numberOfLines = 0
            maskLabelView.textAlignment = .center
            maskLabelView.translatesAutoresizingMaskIntoConstraints = false
            let constraintWidth = NSLayoutConstraint(item: maskLabelView, attribute: .width, relatedBy: .equal, toItem: nil, attribute: .notAnAttribute, multiplier: 1, constant: 300)
            let constraintHeight = NSLayoutConstraint(item: maskLabelView, attribute: .height, relatedBy: .equal, toItem: nil, attribute: .notAnAttribute, multiplier: 1, constant: 300)
            let constraintCenterX = NSLayoutConstraint(item: maskLabelView, attribute: .centerX, relatedBy: .equal, toItem: maskView, attribute: .centerX, multiplier: 1.0, constant: 0.0);
            let constraintCenterY = NSLayoutConstraint(item: maskLabelView, attribute: .centerY, relatedBy: .equal, toItem: maskView, attribute: .centerY, multiplier: 1.0, constant: 0.0);
            maskView.addConstraints([constraintWidth, constraintHeight, constraintCenterX, constraintCenterY])
            self.view.addSubview(maskView)
            return
        }
        
        let modelPath: String
        if gpuModelType == .mobileNet {
            modelPath = Bundle.main.path(forResource: "mobileNetModel", ofType: "json") ?! "can't find mobileNetModel json"
        }else if gpuModelType == .squeezeNet{
            modelPath = Bundle.main.path(forResource: "squeezenet", ofType: "json") ?! "can't find squeezenet json"
        }else{
            fatalError("undefine type")
        }
        
        do{
            let ker: MetalKernel?
            if gpuModelType == .mobileNet {
                ker = MobileNetPreprocessing(device: device)
            }else if gpuModelType == .squeezeNet{
                ker = SqueezeNetPreprocess(device: device)
            }else{
                ker = nil
            }
            
            try MDLGPUNet.share.load(device: device, modelPath: modelPath, preProcessKernel: ker, commandQueue: commandQueue, para: { (matrixName, count) -> NetParameterData? in
                let bundle: Bundle
                if gpuModelType == .mobileNet{
                    let bundlePath = Bundle.main.path(forResource: "MobileNetParameters", ofType: "bundle") ?! "can't find MobileNetParameters.bundle"
                    bundle = Bundle.init(path: bundlePath) ?! "can't load MobileNetParameters.bundle"
                }else if gpuModelType == .squeezeNet{
                    let bundlePath = Bundle.main.path(forResource: "SqueezenetParameters", ofType: "bundle") ?! "can't find SqueezenetParameters.bundle"
                    bundle = Bundle.init(path: bundlePath) ?! "can't load SqueezenetParameters.bundle"
                }else{
                    fatalError("undefine type")
                }
                return NetParameterLoaderBundle(name: matrixName, count: count, ext: "bin", bundle: bundle)
            })
        }catch {
            print(error)
            switch error {
            case NetError.loaderError(message: let message):
                print(message)
            case NetError.modelDataError(message: let message):
                print(message)
            default:
                break
            }
        }
        completion()
    }
    
    func predict(texture: MTLTexture) {
        do {
            try MDLGPUNet.share.predict(inTexture: texture, completion: { (result) in
                self.show(result: result)
            })
        } catch  {
            print(error)
        }
    }
    
    func show(result: MDLNetResult) {
        var s: [String] = ["耗时: \(result.elapsedTime) s"]
        result.result.top(r: 5).enumerated().forEach{
            s.append(String(format: "%d: %@ (%3.2f%%)", $0 + 1, labels[$1.0], $1.1 * 100))
        }
        resultLabel.text = s.joined(separator: "\n\n")
    }
    
    //CaptureTextureDelegate
    func capture(_ texture: MTLTexture!) {
        if let inTexutre = texture {
            predict(texture: inTexutre)
        }
    }
}





