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
import paddle_mobile
import MetalPerformanceShaders

let platform: Platform = .GPU
let threadSupport = [1]

let modelHelperMap: [SupportModel : Runner] = [.mobilenet_ssd : Runner.init(inNet: MobileNet_ssd_hand.init(device: MetalHelper.shared.device), commandQueue: MetalHelper.shared.queue, inPlatform: platform),
                                               .genet : Runner.init(inNet: Genet.init(device: MetalHelper.shared.device), commandQueue: MetalHelper.shared.queue, inPlatform: platform)]
//, .genet : Genet.init()
//let modelHelperMap: [SupportModel : Net] = [.mobilenet : MobileNet.init(), .mobilenet_ssd : MobileNet_ssd_hand.init()]

enum SupportModel: String{
  //  case mobilenet = "mobilenet"
  case mobilenet_ssd = "mobilenetssd"
  case genet          = "genet"
  static func supportedModels() -> [SupportModel] {
    //.mobilenet,
    return [.mobilenet_ssd, .genet]
  }
}

class ViewController: UIViewController {
  @IBOutlet weak var resultTextView: UITextView!
  @IBOutlet weak var selectImageView: UIImageView!
  @IBOutlet weak var elapsedTimeLabel: UILabel!
  @IBOutlet weak var modelPickerView: UIPickerView!
  @IBOutlet weak var threadPickerView: UIPickerView!
  
  var selectImage: UIImage?
  var inputPointer: UnsafeMutablePointer<Float32>?
  var modelType: SupportModel = SupportModel.supportedModels()[0]
  var toPredictTexture: MTLTexture?
  
  var runner: Runner {
    
    get {
      return modelHelperMap[modelType] ?! " has no this type "
    }
    set {
    }
  }
  
  var threadNum = 1
  
  @IBAction func loadAct(_ sender: Any) {
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
    let max = 50
    switch platform {
    case .GPU:
      guard let inTexture = toPredictTexture else {
        resultTextView.text = "请选择图片 ! "
        return
      }
      
      let startDate = Date.init()
      for i in 0..<max {
        runner.predict(texture: inTexture) { [weak self] (success, res) in
          guard let sSelf = self else {
            fatalError()
          }
          if success {
            if i == max - 1 {
              let time = Date.init().timeIntervalSince(startDate)
              DispatchQueue.main.async {
                sSelf.resultTextView.text = sSelf.runner.net.resultStr(res: res)
                sSelf.elapsedTimeLabel.text = "平均耗时: \(time/Double(max) * 1000.0) ms"
              }
            }
          }
        }
      }
      
      
    case .CPU:
      guard let inInputPointer = inputPointer else {
        fatalError( " need input pointer " )
      }
      
      for _ in 0..<10 {
        runner.predict(inputPointer: inInputPointer) { (success, res) in
        }
      }
      
      let startDate = Date.init()
      for i in 0..<max {
        runner.predict(inputPointer: inInputPointer) { [weak self](success, res) in
          guard let sSelf = self else {
            fatalError()
          }
          if success {
            if i == max - 1 {
              let time = Date.init().timeIntervalSince(startDate)
              DispatchQueue.main.async {
                sSelf.resultTextView.text = sSelf.runner.net.resultStr(res: res)
                sSelf.elapsedTimeLabel.text = "平均耗时: \(time/Double(max) * 1000.0) ms"
              }
            }
          }
        }
      }
    }
  }
  
  override func viewDidLoad() {
    super.viewDidLoad()
    modelPickerView.delegate = self
    modelPickerView.dataSource = self
    threadPickerView.delegate = self
    threadPickerView.dataSource = self
    
    selectImage = UIImage.init(named: "hand.jpg")
    selectImageView.image = selectImage
    
    if platform == .CPU {
      inputPointer = runner.preproccess(image: selectImage!.cgImage!)
    } else if platform == .GPU {
      runner.getTexture(image: selectImage!.cgImage!) {[weak self] (texture) in
        self?.toPredictTexture = texture
      }
    } else {
      fatalError( " unsupport " )
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
      return threadSupport.count
    } else {
      fatalError()
    }
  }
  
  public func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
    if pickerView == modelPickerView {
      return SupportModel.supportedModels()[row].rawValue
    } else if pickerView == threadPickerView {
      return "\(threadSupport[row])"
    } else {
      fatalError()
    }
  }
  
  public func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
    if pickerView == modelPickerView {
      self.modelType = SupportModel.supportedModels()[row]
    } else if pickerView == threadPickerView {
      self.threadNum = threadSupport[row]
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


