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

import Foundation
import paddle_mobile

class MobileNet: Net{

  var program: Program?
  
  var executor: Executor<Float32>?
  
  let except: Int = 0
  
  class MobilenetPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = CusomKernel.Shape.init(inWidth: 224, inHeight: 224, inChannel: 3)
      super.init(device: device, inFunctionName: "mobilenet_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  class PreWords {
    var contents: [String] = []
    init(fileName: String, type: String = "txt", inBundle: Bundle = Bundle.main) {
      if let filePath = inBundle.path(forResource: fileName, ofType: type) {
        let string = try! String.init(contentsOfFile: filePath)
        contents = string.components(separatedBy: CharacterSet.newlines).filter{$0.count > 10}.map{
          String($0[$0.index($0.startIndex, offsetBy: 10)...])
        }
      }else{
        fatalError("no file call \(fileName)")
      }
    }
    subscript(index: Int) -> String {
      return contents[index]
    }
  }
  
  let labels = PreWords.init(fileName: "synset")
  
  func resultStr(res: [Float]) -> String {
    var s: [String] = []
    res.top(r: 5).enumerated().forEach{
      s.append(String(format: "%d: %@ (%3.2f%%)", $0 + 1, labels[$1.0], $1.1 * 100))
    }
    return s.joined(separator: "\n")
  }
  
  var preprocessKernel: CusomKernel
  let dim = [1, 224, 224, 3]
  let modelPath: String
  let paramPath: String
  let modelDir: String
  
  init() {
    modelPath = Bundle.main.path(forResource: "model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = MobilenetPreProccess.init(device: MetalHelper.shared.device)
  }
}

