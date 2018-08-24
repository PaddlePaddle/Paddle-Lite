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

struct PriorBoxMetalParam {
    let offset: Float32
    let stepWidth: Float32
    let stepHeight: Float32
    let minSize: Float32
    let maxSize: Float32
    let imageWidth: Float32
    let imageHeight: Float32
    let clip: Bool
    let numPriors: uint
    let aspecRatiosSize: uint
    let minSizeSize: uint
    let maxSizeSize: uint
}

class PriorBoxKernel<P: PrecisionType>: Kernel, Computable{
    var metalParam: PriorBoxMetalParam!
    
    required init(device: MTLDevice, param: PriorBoxParam<P>) {
        super.init(device: device, inFunctionName: "prior_box")
        param.output.initTexture(device: device, transpose: [2, 0, 1, 3])
        param.outputVariances.initTexture(device: device, transpose: [2, 0, 1, 3])
        
        let imageWidth = Float32(param.inputImage.originDim[3])
        let imageHeight = Float32(param.inputImage.originDim[2])
        
        let featureWidth = param.inputImage.originDim[3]
        let featureHeight = param.inputImage.originDim[2]
       
        if param.stepW == 0 || param.stepH == 0 {
            param.stepW = Float32(imageWidth) / Float32(featureWidth)
            param.stepH = Float32(imageHeight) / Float32(featureHeight)
        }
      
        var outputAspectRatior: [Float32] = []
        outputAspectRatior.append(1.0)
        
        let epsilon = 1e-6
        for ar in param.aspectRatios {
            var alreadyExist = false
            for outputAr in outputAspectRatior {
                if fabs(Double(ar) - Double(outputAr)) < Double(epsilon) {
                    alreadyExist = true
                    break
                }
            }
            
            if !alreadyExist {
                outputAspectRatior.append(ar)
            }
            if param.flip {
                outputAspectRatior.append(1.0 / ar)
            }
        }
        
        param.newAspectRatios = outputAspectRatior
        let aspectRatiosSize = uint(outputAspectRatior.count)
                
        let maxSizeSize: uint = uint(param.maxSizes.count)
        let minSizeSize: uint = uint(param.minSizes.count)
        
        let numPriors = aspectRatiosSize * minSizeSize + maxSizeSize
        
        let minSize = param.minSizes.last ?? 0.0
        let maxSize = param.maxSizes.last ?? 0.0
        
        metalParam = PriorBoxMetalParam.init(offset: param.offset, stepWidth: param.stepW, stepHeight: param.stepH, minSize: minSize, maxSize: maxSize, imageWidth: imageWidth, imageHeight: imageHeight, clip: param.clip, numPriors: numPriors, aspecRatiosSize: aspectRatiosSize, minSizeSize: minSizeSize, maxSizeSize: maxSizeSize)
        
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: PriorBoxParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        encoder.setTexture(param.outputVariances.metalTexture, index: 2)
        encoder.setBytes(&metalParam, length: MemoryLayout<PriorBoxMetalParam>.size, index: 0)
        encoder.setBytes(param.aspectRatios, length: MemoryLayout<Float32>.size * param.aspectRatios.count, index: 1)
        encoder.setBytes(param.variances, length: MemoryLayout<Float32>.size * param.variances.count, index: 2)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
   
}
