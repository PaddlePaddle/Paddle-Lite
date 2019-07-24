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

struct SplitMetalParam {
    var idim: (Int32, Int32, Int32, Int32) = (1, 1, 1, 1)
    var axis: Int32 = 0
    var offset: Int32 = 0
    var trans: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
    var vdim: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
}

class SplitKernel<P: PrecisionProtocol>: Kernel, Computable{
    var smp: SplitMetalParam
    func compute(commandBuffer: MTLCommandBuffer, param: SplitParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let inputMetalTexture = param.input.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture is nil")
        }
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(inputMetalTexture, index: 0)
            for i in 0..<param.outputList.count {
                guard let outputMetalTexture = param.outputList[i].metalTexture else {
                    throw PaddleMobileError.makeError(type: .predictError, msg: "output metaltexture \(i) is nil")
                }
                encoder.setTexture(outputMetalTexture, index: i + 1)
            }
            encoder.setBytes(&smp, length: MemoryLayout<SplitMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: inputMetalTexture)
        }
    }
    
    required init(device: MTLDevice, param: SplitParam<P>, initContext: InitContext) throws {
        //     param.output.initTexture(device: device, computePrecision: computePrecision)
        let num = param.outputList.count
        let rank = param.input.tensorDim.cout()
        guard num >= 2 && num <= 4 else {
            throw PaddleMobileError.makeError(type: .netError, msg: "param.outputList.count should satisfy num >= 2 && num <= 4")
        }
        for output in param.outputList {
            try output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        }
        smp = SplitMetalParam.init()
        smp.idim = (Int32(param.input.dim[0]), Int32(param.input.dim[1]), Int32(param.input.dim[2]), Int32(param.input.dim[3]))
        smp.axis = Int32(param.axis + param.input.dim.cout() - param.input.tensorDim.cout())
        for i in 0..<4 {
            if param.input.transpose[i] == smp.axis {
                smp.axis = Int32(i)
                break
            }
        }
        smp.trans = (Int32(param.input.transpose[0]), Int32(param.input.transpose[1]), Int32(param.input.transpose[2]), Int32(param.input.transpose[3]))
        var vdim: [Int32] = [0, 0, 0, 0]
        for i in 0..<num {
            vdim[i] = Int32(param.outputList[i].tensorDim[param.axis])
        }
        smp.vdim = (vdim[0], vdim[1], vdim[2], vdim[3])
        var v = "normal"
        if rank == 4 {
            if smp.axis == 1 {
                v = "y"
            } else if smp.axis == 2 {
                v = "x"
            }
        } else if rank == 3 {
            if smp.axis == 2 {
                v = "y"
            } else if smp.axis == 3 {
                v = "x"
            }
        } else if rank == 2 {
            if smp.axis == 2 {
                v = "y"
            }
        }
        if v == "normal" {
            throw PaddleMobileError.makeError(type: .netError, msg: "unsupported split type")
        }
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "split_\(rank)_\(num)_\(v)_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "split_\(rank)_\(num)_\(v)_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
}
