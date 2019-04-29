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

struct TransposeMetalParam {
    var iC: Int32 = 0
    var oC: Int32 = 0
    var axis: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
}

class TransposeKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: TransposeMetalParam = TransposeMetalParam.init()
    required init(device: MTLDevice, param: TransposeParam<P>, initContext: InitContext) throws {
        
        do {
            try param.output.initTexture(device: device, computePrecision: GlobalConfig.shared.computePrecision)
        } catch let error {
            throw error
        }
        
        let rank = param.input.tensorDim.cout()
        var axis: [Int] = [0, 1, 2, 3]
        for i in 0..<param.axis.count {
            axis[4-rank+i] = 4 - rank + Int(param.axis[i])
        }
        
        var naxis: [Int] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<4 {
                if param.input.transpose[j] == axis[i] {
                    naxis[i] = j
                    break
                }
            }
        }
        metalParam.iC = Int32(param.input.dim[param.input.transpose[3]])
        metalParam.oC = Int32(param.output.dim[3])
        metalParam.axis = (Int32(naxis[0]), Int32(naxis[1]), Int32(naxis[2]), Int32(naxis[3]))
        var kernelFunc = "transpose_undefined"
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.input.transpose == axis {
                kernelFunc = "transpose_copy_half"
            } else {
                kernelFunc = "transpose_\(rank)_half"
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.input.transpose == axis {
                kernelFunc = "transpose_copy_float"
            } else {
                kernelFunc = "transpose_\(rank)_float"
            }
        } else {
            fatalError()
        }
        print("===========>", kernelFunc)
        print(metalParam)
        super.init(device: device, inFunctionName: kernelFunc, initContext: initContext)
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: TransposeParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setBytes(&metalParam, length: MemoryLayout<TransposeMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    
}
