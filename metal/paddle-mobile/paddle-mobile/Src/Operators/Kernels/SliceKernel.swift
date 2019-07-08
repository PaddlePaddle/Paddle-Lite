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

public struct SliceMetalParam {
    let start0: Int16
    let start1: Int16
    let start2: Int16
    let start3: Int16
    let end0: Int16
    let end1: Int16
    let end2: Int16
    let end3: Int16
    let iC: Int32
    let oC: Int32
}

class SliceKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: SliceMetalParam
    var device: MTLDevice?
    var initContext: InitContext?
    
    func compute(commandBuffer: MTLCommandBuffer, param: SliceParam<P>) throws {
        let expectedTranspose = [0, 2, 3, 1]
        var input = param.input
        if param.input.transpose != expectedTranspose {
            if let device = device, let initContext = initContext, let transposedInput = encodeTransposeInput(input: param.input, toTranspose: expectedTranspose, commandBuffer: commandBuffer, device: device, initContext: initContext) {
                input = transposedInput
            } else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "input transpose failed in slice kernel")
            }
        }
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let inputMetalTexture = input.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture is nil")
        }
        guard let outputMetalTexture = param.output.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "output metaltexture is nil")
        }
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(inputMetalTexture, index: 0)
            encoder.setTexture(outputMetalTexture, index: 1)
            encoder.setBytes(&metalParam, length: MemoryLayout<SliceMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
    
    required init(device: MTLDevice, param: SliceParam<P>, initContext: InitContext) throws {
        try param.output.initTexture(device: device, inTranspose: [0, 2, 3, 1], computePrecision: GlobalConfig.shared.computePrecision)
        var ranges = [[Int16]]()
        for i in 0..<4 {
            if let range = param.ranges[i] {
                ranges.append(range)
            } else {
                ranges.append([0, Int16(param.input.tensorDim[i])])
            }
        }
        let start0 = ranges[0][0]
        let start1 = ranges[1][0]
        let start2 = ranges[2][0]
        let start3 = ranges[3][0]
        let end0 = ranges[0][1]
        let end1 = ranges[1][1]
        let end2 = ranges[2][1]
        let end3 = ranges[3][1]
        
        let iC = Int32(param.input.tensorDim[1])
        let oC = Int32(param.output.tensorDim[1])
        
        metalParam = SliceMetalParam.init(start0: start0, start1: start1, start2: start2, start3: start3, end0: end0, end1: end1, end2: end2, end3: end3, iC: iC, oC: oC)
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "slice", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "slice_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
        self.device = device
        self.initContext = initContext
    }
}
