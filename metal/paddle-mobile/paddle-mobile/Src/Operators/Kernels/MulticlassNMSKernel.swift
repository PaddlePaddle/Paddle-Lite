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
            pipline1 = device.pipeLine(funcName: "nms_fetch_bbox", metalLoadMode: initContext.metalLoadMode, metalLibPath: initContext.metalLibPath)
            super.init(device: device, inFunctionName: "nms_fetch_result", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            pipline1 = device.pipeLine(funcName: "nms_fetch_bbox_half", metalLoadMode: initContext.metalLoadMode, metalLibPath: initContext.metalLibPath)
            super.init(device: device, inFunctionName: "nms_fetch_result_half", initContext: initContext)
        } else {
            fatalError( " unsupport precision " )
        }
        
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: MulticlassNMSParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setTexture(param.scores.metalTexture, index: 0)
        encoder.setBuffer(param.middleOutput.resultBuffer!, offset: 0, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.scores.metalTexture)
        encoder.endEncoding()
        
        guard let encoderBox = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoderBox.setTexture(param.bboxes.metalTexture, index: 0)
        encoderBox.setBuffer(param.bboxOutput.resultBuffer!, offset: 0, index: 0)
        encoderBox.dispatch(computePipline: pipline1, outTexture: param.bboxes.metalTexture)
        encoderBox.endEncoding()
    }
}
