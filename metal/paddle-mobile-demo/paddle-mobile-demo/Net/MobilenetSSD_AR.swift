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

public class MobileNet_ssd_AR: Net {
    @objc public override init(device: MTLDevice) throws {
        try super.init(device: device)
        except = 2
        guard let modelPath = Bundle.main.path(forResource: "ar_model", ofType: nil) else {
            throw PaddleMobileError.makeError(type: PaddleMobileErrorType.loaderError, msg: "model null")
        }
        self.modelPath = modelPath
        guard let paramPath = Bundle.main.path(forResource: "ar_params", ofType: nil) else {
            throw PaddleMobileError.makeError(type: PaddleMobileErrorType.loaderError, msg: "para null")
        }
        self.paramPath = paramPath
        preprocessKernel = try MobilenetssdPreProccess.init(device: device)
        inputDim = Dim.init(inDim: [1, 160, 160, 3])
        metalLoadMode = .LoadMetalInCustomMetalLib
        guard let metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib") else {
            throw PaddleMobileError.makeError(type: PaddleMobileErrorType.loaderError, msg: "metallib null")
        }
        self.metalLibPath = metalLibPath
    }
    
    @objc override public init(device: MTLDevice, inParamPointer: UnsafeMutableRawPointer, inParamSize:Int, inModelPointer: UnsafeMutableRawPointer, inModelSize: Int) throws {
        try super.init(device:device,inParamPointer:inParamPointer,inParamSize:inParamSize,inModelPointer:inModelPointer,inModelSize:inModelSize)
        except = 2
        preprocessKernel = try MobilenetssdPreProccess.init(device: device)
        inputDim = Dim.init(inDim: [1, 160, 160, 3])
        metalLoadMode = .LoadMetalInCustomMetalLib
        metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib")
    }
    
    class MobilenetssdPreProccess: CusomKernel {
        init(device: MTLDevice) throws  {
            let s = Shape.init(inWidth: 160, inHeight: 160, inChannel: 3)
            try super.init(device: device, inFunctionName: "mobilent_ar_preprocess", outputDim: s, metalLoadModel: .LoadMetalInDefaultLib, metalLibPath: nil)
        }
    }
    
    override public func resultStr(res: [ResultHolder]) -> String {
        return " \(res[0].result[0])"
    }
    
    override public func fetchResult(paddleMobileRes: [GPUResultHolder]) -> [ResultHolder] {
        return []
    }
}
