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

struct ElementwiseAddMetalParam {
    var fast: Int32 = 0
    var addByChannel: Int32 = 0
    var axis: Int32 = 0
    var ylen: Int32 = 0
    var xdim: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
    var xtrans: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
    var ydim: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
    var ytrans: (Int32, Int32, Int32, Int32) = (0, 1, 2, 3)
}

class ElementwiseAddKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: ElementwiseAddMetalParam
    required init(device: MTLDevice, param: ElementwiseAddParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, inTranspose: param.inputX.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        metalParam = ElementwiseAddKernel.metalParamFrom(inputX: param.inputX, inputY: param.inputY, axis: param.axis)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            try super.init(device: device, inFunctionName: "elementwise_add", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            try super.init(device: device, inFunctionName: "elementwise_add_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ElementwiseAddParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let inputXMetalTexture = param.inputX.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "inputX metaltexture is nil")
        }
        guard let inputYMetalTexture = param.inputY.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "inputY metaltexture is nil")
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
            encoder.setTexture(inputXMetalTexture, index: 0)
            encoder.setTexture(inputYMetalTexture, index: 1)
            encoder.setTexture(outputMetalTexture, index: 2)
            encoder.setBytes(&metalParam, length: MemoryLayout<ElementwiseAddMetalParam>.size, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
    
    static func metalParamFrom(inputX: Texture, inputY: Texture, axis: Int) -> ElementwiseAddMetalParam {
        var metalParam = ElementwiseAddMetalParam.init()
        
        let xdim: [Int32] = (0..<4).map { Int32(inputX.dim[$0]) }
        let ydim: [Int32] = (0..<4).map { Int32(inputY.dim[$0]) }
        let xtrans: [Int32] = (0..<4).map { Int32(inputX.transpose[$0]) }
        let ytrans: [Int32] = (0..<4).map { Int32(inputY.transpose[$0]) }
        
        metalParam.xdim = (xdim[0], xdim[1], xdim[2], xdim[3])
        metalParam.ydim = (ydim[0], ydim[1], ydim[2], ydim[3])
        metalParam.xtrans = (xtrans[0], xtrans[1], xtrans[2], xtrans[3])
        metalParam.ytrans = (ytrans[0], ytrans[1], ytrans[2], ytrans[3])
        if axis == -1 {
            metalParam.axis = 4 - Int32(inputY.tensorDim.cout())
        } else {
            metalParam.axis = 4 - Int32(inputX.tensorDim.cout()) + Int32(axis)
        }
        metalParam.ylen = Int32(inputY.tensorDim.cout())
        if (inputX.dim == inputY.dim) && (inputX.transpose == inputY.transpose) {
            //      print("===> elementwise_add fast!!!")
            metalParam.fast = 1
        }
        if inputY.tensorDim.cout() == 1 && (axis == 1 || (axis == -1 && inputY.tensorDim.dims[0] == inputX.padToFourDim[1])) {
            metalParam.addByChannel = 1
        }
        return metalParam
    }
}
