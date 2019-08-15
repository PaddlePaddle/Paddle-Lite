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

struct MetalConvTransposeParam {
    let kernelW: UInt16;
    let kernelH: UInt16;
    
    let strideX: UInt16;
    let strideY: UInt16;
    
    let paddingX: UInt16;
    let paddingY: UInt16;
    
    let dilationX: UInt16;
    let dilationY: UInt16;
}

class ConvTransposeKernel<P: PrecisionProtocol>: Kernel, Computable{
    var metalParam: MetalConvTransposeParam!
    required init(device: MTLDevice, param: ConvTransposeParam<P>, initContext: InitContext) throws {
        try param.output.initTexture(device: device, inTranspose: param.input.transpose, computePrecision: GlobalConfig.shared.computePrecision)
        
        try param.filter.initBuffer(device: device, precision: GlobalConfig.shared.computePrecision, convertToNHWC: false, withTranspose: true)
        if GlobalConfig.shared.computePrecision == .Float32 {
            if param.stride == [2, 2] && param.stride == [2, 2] {
                try super.init(device: device, inFunctionName: "conv_transpose2x2_stride2", initContext: initContext)
            } else {
                throw PaddleMobileError.makeError(type: .netError, msg: "conv transpose param \(param) unsupported yet")
            }
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            if param.stride == [2, 2] && param.stride == [2, 2] {
                try super.init(device: device, inFunctionName: "conv_transpose2x2_stride2_half", initContext: initContext)
            } else {
                throw PaddleMobileError.makeError(type: .netError, msg: "conv transpose param \(param) unsupported yet")
            }
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
        
        //    let filter: [Float32] = param.filter.buffer.array()
        //    print(" conv transpose filter")
        //    print(filter)
        guard let filterWidth = param.filter.width else {
            throw PaddleMobileError.makeError(type: .netError, msg: "filter unsupported")
        }
        guard let filterHeight = param.filter.height else {
            throw PaddleMobileError.makeError(type: .netError, msg: "filter unsupported")
        }
        let kernelWidth = UInt16(filterWidth)
        let kernelHeight = UInt16(filterHeight)
        
        let strideX = UInt16(param.stride[0])
        let strideY = UInt16(param.stride[1])
        let paddingX = UInt16(param.paddings[0])
        let paddingY = UInt16(param.paddings[1])
        let dilationX = UInt16(param.dilations[0])
        let dilationY = UInt16(param.dilations[1])
        
        metalParam = MetalConvTransposeParam.init(kernelW: kernelWidth, kernelH: kernelHeight, strideX: strideX, strideY: strideY, paddingX: paddingX, paddingY: paddingY, dilationX: dilationX, dilationY: dilationY)
        
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvTransposeParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let inputMetalTexture = param.input.metalTexture else {
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
            encoder.setBytes(&metalParam, length: MemoryLayout<MetalConvTransposeParam>.size, index: 0)
            encoder.setBuffer(param.filter.buffer, offset: 0, index: 1)
            try encoder.dispatch(computePipline: tempPipline, outTexture: outputMetalTexture)
        }
    }
}


