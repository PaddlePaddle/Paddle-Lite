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

import Metal
import Foundation
import paddle_mobile

public class YoloNet: Net {
    @objc public override init(device: MTLDevice) throws {
        try super.init(device: device)
        except = 0
        guard let modelPath = Bundle.main.path(forResource: "yolo_16_model", ofType: nil) else {
            throw PaddleMobileError.makeError(type: PaddleMobileErrorType.loaderError, msg: "model null")
        }
        self.modelPath = modelPath
        guard let paramPath = Bundle.main.path(forResource: "yolo_16_param", ofType: nil) else {
            throw PaddleMobileError.makeError(type: PaddleMobileErrorType.loaderError, msg: "para null")
        }
        self.paramPath = paramPath
        inputDim = Dim.init(inDim: [1, 416, 416, 3])
        metalLoadMode = .LoadMetalInCustomMetalLib
        guard let metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib") else {
            throw PaddleMobileError.makeError(type: PaddleMobileErrorType.loaderError, msg: "metallib null")
        }
        self.metalLibPath = metalLibPath
        useMPS = true
        paramPrecision = .Float16
        preprocessKernel = try ScaleKernel.init(device: device, shape: Shape.init(inWidth: 416, inHeight: 416, inChannel: 3), metalLoadMode: .LoadMetalInCustomMetalLib, metalLibPath: Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib"))
        
    }
    
    override  public func resultStr(res: [ResultHolder]) -> String {
        return " \(res[0].result[0]) ... "
    }
    
}
