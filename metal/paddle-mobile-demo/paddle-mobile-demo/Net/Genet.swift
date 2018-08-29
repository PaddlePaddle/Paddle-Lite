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

class Genet: Net {
  
  var program: Program?
  
  var executor: Executor<Float32>?
  
  let except: Int = 0
  
  class GenetPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = CusomKernel.Shape.init(inWidth: 128, inHeight: 128, inChannel: 3)
      super.init(device: device, inFunctionName: "genet_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  func resultStr(res: [Float]) -> String {
    return " 哈哈  还没好 genet !";
  }
  
  var preprocessKernel: CusomKernel
  let dim = (n: 1, h: 128, w: 128, c: 3)
  let modelPath: String
  let paramPath: String
  let modelDir: String
  
  init() {
    modelPath = Bundle.main.path(forResource: "genet_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "genet_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = GenetPreProccess.init(device: MetalHelper.shared.device)
  }
  
}
