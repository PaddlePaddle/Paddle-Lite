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
import Metal

struct ConcatTestParam: TestParam {
    var input: [MTLTexture]
    var output: MTLTexture
    var dims: [[Int]]
    var axis: Int
    var odim: [Int]
}

struct ConcatMetalParam {
    var odim: (Int32, Int32, Int32, Int32) = (1, 1, 1, 1)
    var axis: Int32 = 0
    var offset: Int32 = 0
    var trans: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
    var vdim: (Int32, Int32, Int32, Int32, Int32, Int32) = (0, 0, 0, 0, 0, 0)
}

class ConcatKernel<P: PrecisionProtocol>: Kernel, Computable{
    var v = "normal"
    var pm = ConcatMetalParam.init()
    func compute(commandBuffer: MTLCommandBuffer, param: ConcatParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
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
            let num = param.input.count
            for i in 0..<num {
                guard let inputMetalTexture = param.input[i].metalTexture else {
                    throw PaddleMobileError.makeError(type: .predictError, msg: "input metaltexture \(i) is nil")
                }
                encoder.setTexture(inputMetalTexture, index: i)
            }
            encoder.setTexture(outputMetalTexture, index: num)
            if v == "normal" {
                encoder.setTexture(param.output.metalTexture, index: num + 1)
            }
            encoder.setBytes(&pm, length: MemoryLayout<ConcatMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
    
    required init(device: MTLDevice, param: ConcatParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, inTranspose: param.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        let orank = param.output.tensorDim.cout()
        let num = param.input.count
        guard num <= 6 else {
            throw PaddleMobileError.makeError(type: .netError, msg: "param input count must be less than or equal to 6")
        }
        var axis = 4 - param.output.tensorDim.cout() + param.axis
        for i in 0..<4 {
            if param.transpose[i] == axis {
                axis = i
                break
            }
        }
        pm.axis = Int32(axis)
        pm.odim = (Int32(param.output.dim[0]), Int32(param.output.dim[1]), Int32(param.output.dim[2]), Int32(param.output.dim[3]))
        pm.trans = (Int32(param.output.transpose[0]), Int32(param.output.transpose[1]), Int32(param.output.transpose[2]), Int32(param.output.transpose[3]))
        var vdim: [Int] = [0, 0, 0, 0, 0, 0]
        for i in 0..<num {
            vdim[i] = param.input[i].dim[axis]
        }
        if orank == 4 {
            if axis == 1 {
                v = "y"
            } else if axis == 2 {
                v = "x"
            } else {
                if (param.output.dim[0] == 1) && axis == 3 {
                    var vz = true
                    for i in 0..<num {
                        if vdim[i] % 4 != 0 {
                            vz = false
                            break
                        }
                    }
                    if vz {
                        v = "z"
                        for i in 0..<num {
                            vdim[i] = vdim[i] / 4
                        }
                    }
                }
            }
        } else if orank == 3 {
            if axis == 2 {
                v = "y"
            } else if axis == 3 {
                v = "x"
            } else if axis == 1 {
                var vz = true
                for i in 0..<num {
                    if vdim[i] % 4 != 0 {
                        vz = false
                        break
                    }
                }
                if vz {
                    v = "z"
                    for i in 0..<num {
                        vdim[i] = vdim[i] / 4
                    }
                }
            }
        } else {
            if axis == 2 {
                v = "y"
            } else if axis == 3 {
                var vx = true
                for i in 0..<num {
                    if vdim[i] % 4 != 0 {
                        vx = false
                        break
                    }
                }
                if vx {
                    v = "x"
                    for i in 0..<num {
                        vdim[i] = vdim[i] / 4
                    }
                }
            }
        }
        pm.vdim = (Int32(vdim[0]), Int32(vdim[1]), Int32(vdim[2]), Int32(vdim[3]), Int32(vdim[4]), Int32(vdim[5]))
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "concat_\(orank)_\(num)_\(v)_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "concat_\(orank)_\(num)_\(v)_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    required init(device: MTLDevice, testParam: ConcatTestParam, initContext: InitContext) throws {
        try super.init(device: device, inFunctionName: "concat", initContext: initContext)
    }
}
