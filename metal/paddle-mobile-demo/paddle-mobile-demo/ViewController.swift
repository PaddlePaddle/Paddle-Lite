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

let threadSupport = [1]

let modelHelperMap: [SupportModel : Net] = [.mobilenet_ssd : MobileNet_ssd_hand.init(), .genet : Genet.init()]
//let modelHelperMap: [SupportModel : Net] = [.mobilenet : MobileNet.init(), .mobilenet_ssd : MobileNet_ssd_hand.init()]

enum SupportModel: String{
  //  case mobilenet = "mobilenet"
  case mobilenet_ssd = "mobilenetssd"
  case genet          = "enet"
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
  var modelType: SupportModel = SupportModel.supportedModels()[0]
  var toPredictTexture: MTLTexture?
  
  var net: Net {
    get {
      return modelHelperMap[modelType] ?! " has no this type "
    }
    set {
    }
  }
  
  var threadNum = 1
  
  @IBAction func loadAct(_ sender: Any) {
    
    do {
      try self.net.load()
    } catch let error {
      print(error)
    }
  }
  
  @IBAction func selectImageAct(_ sender: Any) {
    let imagePicker = UIImagePickerController()
    imagePicker.sourceType = .camera
    imagePicker.delegate = self
    self.present(imagePicker, animated: true, completion: nil)
  }
  
  @IBAction func clearAct(_ sender: Any) {
    net.clear()
  }
  
  @IBAction func predictAct(_ sender: Any) {
    guard let inTexture = toPredictTexture else {
      resultTextView.text = "请选择图片 ! "
      return
    }
    do {
      let max = 1
      let startDate = Date.init()
      for i in 0..<max {
        try net.predict(inTexture: inTexture) { [weak self] (result) in
          guard let sSelf = self else {
            fatalError()
          }
          
          if i == max - 1 {
            let time = Date.init().timeIntervalSince(startDate)
            DispatchQueue.main.async {
              sSelf.resultTextView.text = sSelf.net.resultStr(res: result.resultArray)
              sSelf.elapsedTimeLabel.text = "平均耗时: \(time/Double(max) * 1000.0) ms"
            }
          }
        }
      }
    } catch let error {
      print(error)
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
    net.getTexture(image: selectImage!.cgImage!) {[weak self] (texture) in
      self?.toPredictTexture = texture
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
      sSelf.net.getTexture(image: image.cgImage!, getTexture: { (texture) in
        sSelf.toPredictTexture = texture
      })
    }
  }
}


