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

import UIKit
import MetalKit
import CoreMedia
import paddle_mobile
import MetalPerformanceShaders

class FileReader {
    let file: UnsafeMutablePointer<FILE>
    let fileSize: Int
    init(paramPath: String) throws {
        guard let tmpFile = fopen(paramPath, "rb") else {
            throw PaddleMobileError.loaderError(message: "open param file error" + paramPath)
        }
        file = tmpFile
        fseek(file, 0, SEEK_END)
        fileSize = ftell(file)
        guard fileSize > 0 else {
            throw PaddleMobileError.loaderError(message: "param file size is too small")
        }
        rewind(file)
    }
    
    func read<T>() -> UnsafeMutablePointer<T> {
        let ptr = UnsafeMutablePointer<T>.allocate(capacity: MemoryLayout<T>.size * fileSize)
        fread(ptr, fileSize, 1, file)
        return ptr
    }
    
    deinit {
        fclose(file)
    }
}

enum Platform {
    case GPU
}

let platformSupport: [(Platform, String)] = [(.GPU, "GPU")]

enum SupportModel: String{
    case yolo               = "yolo"
    case mobilenet_combined = "mobilenet_combined"
    case super_resolution   = "superresoltion"
    case mobilenet          = "mobilenet"
    
    static func supportedModels() -> [SupportModel] {
        return [.super_resolution, .yolo, .mobilenet_combined, .mobilenet]
    }
}

let netSupport: [SupportModel : Net] = [
    .super_resolution : SuperResolutionNet.init(device: MetalHelper.shared.device),
    .yolo : YoloNet.init(device: MetalHelper.shared.device),
    .mobilenet_combined : MobileNetCombined.init(device: MetalHelper.shared.device),
    .mobilenet : MobileNet.init(device: MetalHelper.shared.device)]

class ViewController: UIViewController {
    @IBOutlet weak var resultTextView: UITextView!
    @IBOutlet weak var selectImageView: UIImageView!
    @IBOutlet weak var elapsedTimeLabel: UILabel!
    @IBOutlet weak var modelPickerView: UIPickerView!
    @IBOutlet weak var threadPickerView: UIPickerView!
    @IBOutlet weak var videoView: UIView!
    //  var videoCapture: VideoCapture!
    
    var selectImage: UIImage?
    var inputPointer: UnsafeMutablePointer<Float32>?
    var modelType: SupportModel = SupportModel.supportedModels()[0]
    var toPredictTexture: MTLTexture?
    
    var runner: Runner!
    var platform: Platform = .GPU
    var threadNum = 1
    
    @IBAction func loadAct(_ sender: Any) {
        runner = Runner.init(inNet: netSupport[modelType]!, commandQueue: MetalHelper.shared.queue)
        if platform == .GPU {
            //      let filePath = Bundle.main.path(forResource: "mingren_input_data", ofType: nil)
            //      let fileReader = try! FileReader.init(paramPath: filePath!)
            //      let pointer: UnsafeMutablePointer<Float32> = fileReader.read()
            //      
            //      
            //      let buffer = MetalHelper.shared.device.makeBuffer(length: fileReader.fileSize, options: .storageModeShared)
            //      
            //      buffer?.contents().copyMemory(from: pointer, byteCount: fileReader.fileSize)
            
            
            if self.toPredictTexture == nil {
                let beforeDate = Date.init()
                if modelType == .mobilenet_combined || modelType == .yolo {
                    self.toPredictTexture = try! MetalHelper.shared.textureLoader.newTexture(cgImage: selectImage!.cgImage!, options: nil)
                } else {
                    runner.getTexture(image: selectImage!.cgImage!) { [weak self] (texture) in
                        let timeUse = Date.init().timeIntervalSince(beforeDate)
                        print("get texture time use: \(timeUse)")
                        self?.toPredictTexture = texture
                    }
                }
            }
        } else {
            fatalError( " unsupport " )
        }
        
        if runner.load() {
            print(" load success ! ")
        } else {
            print(" load error ! ")
        }
    }
    
    @IBAction func selectImageAct(_ sender: Any) {
        let imagePicker = UIImagePickerController()
        imagePicker.sourceType = .camera
        imagePicker.delegate = self
        self.present(imagePicker, animated: true, completion: nil)
    }
    
    @IBAction func clearAct(_ sender: Any) {
        runner.clear()
    }
    
    @IBAction func predictAct(_ sender: Any) {
        let max = 1
        switch platform {
        case .GPU:
            guard let inTexture = toPredictTexture else {
                resultTextView.text = "请选择图片 ! "
                return
            }
            
            let startDate = Date.init()
            for i in 0..<max {
                self.runner.predict(texture: inTexture) { [weak self] (success, resultHolder)  in
                    guard let sSelf = self else {
                        fatalError()
                    }
                    
                    if success, let inResultHolderArr = resultHolder {
                        let inResultHolder = inResultHolderArr[0]
                        if i == max - 1 {
                            let time = Date.init().timeIntervalSince(startDate)
                            
                            print(inResultHolder.result.floatArr(count: inResultHolder.capacity).strideArray())
                            DispatchQueue.main.async {
                                sSelf.resultTextView.text = sSelf.runner.net.resultStr(res: resultHolder!)
                                sSelf.elapsedTimeLabel.text = "平均耗时: \(time/Double(max) * 1000.0) ms"
                            }
                        }
                    }
                    
                    DispatchQueue.main.async {
                        resultHolder?.first?.releasePointer()
                    }
                }
            }
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        GlobalConfig.shared.computePrecision = .Float16
        GlobalConfig.shared.debug = false
        
        modelPickerView.delegate = self
        modelPickerView.dataSource = self
        threadPickerView.delegate = self
        threadPickerView.dataSource = self
        if let image = UIImage.init(named: "00001.jpg") {
            selectImage = image
            selectImageView.image = image
        } else {
            print("请添加测试图片")
        }
    }
}

extension ViewController: UIPickerViewDataSource, UIPickerViewDelegate{
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        if pickerView == modelPickerView {
            return 1
        } else if pickerView == threadPickerView {
            return 1
        } else {
            fatalError()
        }
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        if pickerView == modelPickerView {
            return SupportModel.supportedModels().count
        } else if pickerView == threadPickerView {
            return platformSupport.count
        } else {
            fatalError()
        }
    }
    
    public func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        if pickerView == modelPickerView {
            return SupportModel.supportedModels()[row].rawValue
        } else if pickerView == threadPickerView {
            return platformSupport[row].1
        } else {
            fatalError()
        }
    }
    
    public func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        if pickerView == modelPickerView {
            self.modelType = SupportModel.supportedModels()[row]
        } else if pickerView == threadPickerView {
            platform = platformSupport[row].0
        } else {
            fatalError()
        }
    }
}

extension ViewController:  UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true){[weak self] in
            guard let sSelf = self, let image =  info["UIImagePickerControllerOriginalImage"] as? UIImage else{
                fatalError("no image")
            }
            sSelf.selectImage = image
            sSelf.selectImageView.image = image
            sSelf.runner.getTexture(image: image.cgImage!, getTexture: { (texture) in
                sSelf.toPredictTexture = texture
            })
        }
    }
}

var bool1 = false
extension ViewController: VideoCaptureDelegate{
    func predictTexture(texture: MTLTexture){
        runner.scaleTexture(input: texture) { (scaledTexture) in
            self.runner.predict(texture: scaledTexture, completion: { (success, resultHolder) in
                resultHolder?.first?.releasePointer()
            })
        }
    }
    
}




