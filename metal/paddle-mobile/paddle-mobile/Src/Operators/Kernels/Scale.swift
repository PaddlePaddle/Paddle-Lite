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

public class ScaleKernel: CusomKernel {
    public init(device: MTLDevice, shape: Shape, metalLoadMode: MetalLoadMode, metalLibPath: String?) {
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "scale", outputDim: shape, metalLoadModel: metalLoadMode, metalLibPath: metalLibPath)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "scale_half", outputDim: shape, metalLoadModel: metalLoadMode, metalLibPath: metalLibPath)
        } else {
            fatalError(" unsupport ")
        }
    }
}

