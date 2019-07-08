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

class MulticlassNMSKernel<P: PrecisionProtocol>: Kernel, Computable{
    let pipline1: MTLComputePipelineState
    
    required init(device: MTLDevice, param: MulticlassNMSParam<P>, initContext: InitContext) throws {
        
        param.middleOutput.initBuffer(device: device)
        param.bboxOutput.initBuffer(device: device)
        if GlobalConfig.shared.computePrecision == .Float32 {
            pipline1 = try device.pipeLine(funcName: "nms_fetch_bbox", metalLoadMode: initContext.metalLoadMode, metalLibPath: initContext.metalLibPath)
            try super.init(device: device, inFunctionName: "nms_fetch_result", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            pipline1 = try device.pipeLine(funcName: "nms_fetch_bbox_half", metalLoadMode: initContext.metalLoadMode, metalLibPath: initContext.metalLibPath)
            try super.init(device: device, inFunctionName: "nms_fetch_result_half", initContext: initContext)
        } else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "unsupported compute precision: \(GlobalConfig.shared.computePrecision)")
        }
        
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: MulticlassNMSParam<P>) throws {
        guard let tempPipline = pipline else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "pipline is nil")
        }
        guard let scoresMetalTexture = param.scores.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "scores metaltexture is nil")
        }
        guard let bboxesMetalTexture = param.bboxes.metalTexture else {
            throw PaddleMobileError.makeError(type: .predictError, msg: "bboxes metaltexture is nil")
        }
        do {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoder.endEncoding()
            }
            encoder.setTexture(scoresMetalTexture, index: 0)
            encoder.setBuffer(param.middleOutput.resultBuffer!, offset: 0, index: 0)
            try encoder.dispatch(computePipline: tempPipline, outTexture: scoresMetalTexture)
        }
        
        do {
            guard let encoderBox = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.makeError(type: .predictError, msg: "encoder is nil")
            }
            defer {
                encoderBox.endEncoding()
            }
            encoderBox.setTexture(param.bboxes.metalTexture, index: 0)
            encoderBox.setBuffer(param.bboxOutput.resultBuffer!, offset: 0, index: 0)
            try encoderBox.dispatch(computePipline: pipline1, outTexture: bboxesMetalTexture)
        }
    }
}
