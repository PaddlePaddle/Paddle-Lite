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

public class Genet: Net {
    @objc public override init(device: MTLDevice) throws {
        try super.init(device: device)
        modelPath = Bundle.main.path(forResource: "genet_model", ofType: nil) ?! "model null"
        paramPath = Bundle.main.path(forResource: "genet_params", ofType: nil) ?! "para null"
        preprocessKernel = try GenetPreProccess.init(device: device)
        inputDim = Dim.init(inDim: [1, 128, 128, 3])
        metalLoadMode = .LoadMetalInCustomMetalLib
        metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib")
    }
    
    @objc override public init(device: MTLDevice, inParamPointer: UnsafeMutableRawPointer, inParamSize:Int, inModelPointer: UnsafeMutableRawPointer, inModelSize: Int) throws {
        try super.init(device: device,
                   inParamPointer: inParamPointer,
                   inParamSize: inParamSize,
                   inModelPointer: inModelPointer,
                   inModelSize: inModelSize)
        metalLoadMode = .LoadMetalInCustomMetalLib
        metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib")
        preprocessKernel = try GenetPreProccess.init(device: device)
        inputDim = Dim.init(inDim: [1, 128, 128, 3])
    }
    
    class GenetPreProccess: CusomKernel {
        init(device: MTLDevice) throws {
            let s = Shape.init(inWidth: 128, inHeight: 128, inChannel: 3)
            try super.init(device: device, inFunctionName: "genet_preprocess", outputDim: s, metalLoadModel: .LoadMetalInDefaultLib, metalLibPath: nil)
        }
    }
    
    override  public func resultStr(res: [ResultHolder]) -> String {
        return " \(res[0].result[0]) ... "
    }
    
}
