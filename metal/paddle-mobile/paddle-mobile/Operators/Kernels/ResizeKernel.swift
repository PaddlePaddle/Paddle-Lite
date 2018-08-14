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
//
//import Foundation
//import MetalPerformanceShaders
//
//
//struct ResizeParam: OpParam{
//    typealias OutputType = <#type#>
//    
//    typealias ParamPrecisionType = <#type#>
//    
//    let input: MTLTexture
//    let output: MTLTexture
//    let expectDim: Dim
//}
//
//struct OutputDim {
//    let width: UInt16
//    let height: UInt16
//    let strideX: UInt16
//    let strideY: UInt16
//}
//
//class ResizeKernel<P: PrecisionType>: Kernel, Computable{
//    var lanczos: MPSImageLanczosScale
//    required init(device: MTLDevice, param: ResizeParam) {
//        lanczos = MPSImageLanczosScale.init(device: device)
//        super.init(device: device, inFunctionName: "resize")
//    }
//    func compute(commandBuffer: MTLCommandBuffer, param: ResizeParam) throws {
////        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
////            throw PaddleMobileError.predictError(message: " encode is nil")
////        }
//        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: param.input, destinationTexture: param.output)
//        
////        encoder.setTexture(param.input, index: 0)
////        encoder.setTexture(param.output, index: 1)
////        let strideX = param.input.width/param.expectDim[2]
////        let strideY = param.input.height/param.expectDim[1]
////        var outputDim = OutputDim.init(width: UInt16(param.expectDim[1]), height: UInt16(param.expectDim[2]), strideX: UInt16(strideX), strideY: UInt16(strideY))
////        encoder.setBytes(&outputDim, length: MemoryLayout<OutputDim>.size, index: 0)
////        encoder.dispatch(computePipline: pipline, outTexture: param.output)
////        encoder.endEncoding()
//    }
//    
//
//    
//    
//}

