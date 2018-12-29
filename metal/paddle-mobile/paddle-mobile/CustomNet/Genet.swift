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

public class Genet: Net {
  @objc public override init(device: MTLDevice) {
    super.init(device: device)
    means = [128.0, 128.0, 128.0]
    scale = 0.017
    except = 0
    modelPath = Bundle.main.path(forResource: "genet_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "genet_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = GenetPreProccess.init(device: device)
    inputDim_ = Dim.init(inDim: [1, 128, 128, 3])
  }
  
  @objc override public init(device: MTLDevice,paramPointer: UnsafeMutableRawPointer, paramSize:Int, modePointer: UnsafeMutableRawPointer, modelSize: Int) {
    super.init(device:device,paramPointer:paramPointer,paramSize:paramSize,modePointer:modePointer,modelSize:modelSize)
    means = [128.0, 128.0, 128.0]
    scale = 0.017
    except = 0
    modelPath = ""
    paramPath = ""
    modelDir = ""
    preprocessKernel = GenetPreProccess.init(device: device)
    inputDim_ = Dim.init(inDim: [1, 128, 128, 3])
  }

  class GenetPreProccess: CusomKernel {
    init(device: MTLDevice) {
      let s = Shape.init(inWidth: 128, inHeight: 128, inChannel: 3)
      super.init(device: device, inFunctionName: "genet_preprocess", outputDim: s, usePaddleMobileLib: false)
    }
  }
  
  override  public func resultStr(res: ResultHolder) -> String {
//    fatalError()
    return " \(res.result[0]) ... "
  }
  
}
