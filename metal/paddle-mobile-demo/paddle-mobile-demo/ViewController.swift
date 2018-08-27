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

class ViewController: UIViewController {
  @IBOutlet weak var resultTextView: UITextView!
  @IBOutlet weak var selectImageView: UIImageView!
  @IBOutlet weak var elapsedTimeLabel: UILabel!
  @IBOutlet weak var modelPickerView: UIPickerView!
  @IBOutlet weak var threadPickerView: UIPickerView!
  var selectImage: UIImage?
  var program: Program?
  var executor: Executor<Float32>?
  var modelType: SupportModel = SupportModel.supportedModels()[0]
  var toPredictTexture: MTLTexture?
  var modelHelper: Net {
    return modelHelperMap[modelType] ?! " has no this type "
  }
  var threadNum = 1
  
  @IBAction func loadAct(_ sender: Any) {
    let inModelHelper = modelHelper
    let queue = MetalHelper.shared.queue
    let loader = Loader<Float32>.init()
    do {
      let modelPath = inModelHelper.modelPath
      let paraPath = inModelHelper.paramPath
      
      program = try loader.load(device: MetalHelper.shared.device, modelPath: modelPath, paraPath: paraPath)
      executor = try Executor<Float32>.init(inDevice: MetalHelper.shared.device, inQueue: queue, inProgram: program!)
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
    executor?.clear()
    program = nil
    executor = nil
  }
  
  @IBAction func predictAct(_ sender: Any) {
    guard let inTexture = toPredictTexture else {
      resultTextView.text = "请选择图片 ! "
      return
    }
    
    guard let inExecutor = executor else {
      resultTextView.text = "请先 load ! "
      return
    }
    
    do {
      let max = 1
      var startDate = Date.init()
      for i in 0..<max {
        try inExecutor.predict(input: inTexture, expect: modelHelper.dim, completionHandle: { [weak self] (result) in
          guard let sSelf = self else {
            fatalError()
          }
          
          if i == (max / 2 - 1) {
            startDate = Date.init()
          }
          
          let resultArr = sSelf.modelHelper.fetchResult(paddleMobileRes: result)
          
          if i == max - 1 {
            let time = Date.init().timeIntervalSince(startDate)
            DispatchQueue.main.async {
              sSelf.resultTextView.text = sSelf.modelHelper.resultStr(res: resultArr)
              sSelf.elapsedTimeLabel.text = "平均耗时: \(time/Double(max/2) * 1000.0) ms"
            }
          }
          }, preProcessKernle: self.modelHelper.preprocessKernel, except: 2)
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
    modelHelper.getTexture(image: selectImage!.cgImage!) {[weak self] (texture) in
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
      sSelf.modelHelper.getTexture(image: image.cgImage!, getTexture: { (texture) in
        sSelf.toPredictTexture = texture
      })
    }
  }
}


