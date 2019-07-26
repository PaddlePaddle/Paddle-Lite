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


class ElementwiseAddPreluKernel<P: PrecisionProtocol>: Kernel, Computable {
    var metalParam: ElementwiseAddMetalParam
    required init(device: MTLDevice, param: ElementwiseAddPreluParam<P>, initContext: InitContext) throws {
        
        try param.output.initTexture(device: device, inTranspose: param.inputX.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        try param.alpha.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision)
        
        metalParam = ElementwiseAddKernel<P>.metalParamFrom(inputX: param.inputX, inputY: param.inputY, axis: param.axis)
        
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.mode == "channel" {
                try super.init(device: device, inFunctionName: "elementwise_add_channel_float", initContext: initContext)
            } else if param.mode == "element" {
                try super.init(device: device, inFunctionName: "elementwise_add_element_float", initContext: initContext)
            } else {
                try super.init(device: device, inFunctionName: "elementwise_add_prelu_float", initContext: initContext)
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.mode == "channel" {
                try super.init(device: device, inFunctionName: "elementwise_add_channel_half", initContext: initContext)
            } else if param.mode == "element" {
                try super.init(device: device, inFunctionName: "elementwise_add_channel_half", initContext: initContext)
            } else {
                try super.init(device: device, inFunctionName: "elementwise_add_channel_half", initContext: initContext)
            }
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ElementwiseAddPreluParam<P>) throws {
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
            encoder.setBuffer(param.alpha.buffer, offset: 0, index: 1)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
}
