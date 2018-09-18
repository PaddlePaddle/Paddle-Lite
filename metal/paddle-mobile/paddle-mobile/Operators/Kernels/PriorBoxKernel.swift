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
    
    let originDim = param.output.tensorDim;
    
    param.output.tensorDim = Dim.init(inDim: [1, originDim[0], originDim[1], originDim[2] * originDim[3]])
    param.output.padToFourDim = Dim.init(inDim: [1, originDim[0], originDim[1], originDim[2] * originDim[3]])
    
    param.output.initTexture(device: device, inTranspose: [0, 1, 2, 3], computePrecision: computePrecision)
    param.outputVariances.initTexture(device: device, inTranspose: [2, 0, 1, 3], computePrecision: computePrecision)
    
    
    if computePrecision == .Float32 {
      if param.min_max_aspect_ratios_order {
        super.init(device: device, inFunctionName: "prior_box_MinMaxAspectRatiosOrder")
      } else {
        super.init(device: device, inFunctionName: "prior_box")
      }
      
    } else if computePrecision == .Float16 {
      if param.min_max_aspect_ratios_order {
        super.init(device: device, inFunctionName: "prior_box_MinMaxAspectRatiosOrder_half")
      } else {
        super.init(device: device, inFunctionName: "prior_box_half")
      }
    } else {
      fatalError()
    }
    
    
    guard param.minSizes.count == 1 else {
      fatalError(" need implement ")
    }
    
//    let n = 1
//    let h = param.output.dim[1]
//    let w = param.output.dim[2]
//    let c = param.output.dim[3] * param.output.dim[0]
//
//    param.output.dim = Dim.init(inDim: [n, h, w, c])
//    param.output.transpose = [0, 1, 2, 3]
    
    let imageWidth = Float32(param.inputImage.padToFourDim[3])
    let imageHeight = Float32(param.inputImage.padToFourDim[2])
    
    let featureWidth = param.input.padToFourDim[3]
    let featureHeight = param.input.padToFourDim[2]
    
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
    
    if computePrecision == .Float16 {
      let buffer = device.makeBuffer(length: outputAspectRatior.count * MemoryLayout<Float16>.size)
      float32ToFloat16(input: &outputAspectRatior, output:(buffer?.contents())!, count: outputAspectRatior.count)
      param.newAspectRatios = buffer

    } else if computePrecision == .Float32 {
      let buffer = device.makeBuffer(bytes: outputAspectRatior, length: outputAspectRatior.count * MemoryLayout<Float32>.size, options: [])
      param.newAspectRatios = buffer
    } else {
      fatalError()
    }
    
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
    
    encoder.setBuffer(param.newAspectRatios!, offset: 0, index: 0)
    
    encoder.setBytes(&metalParam, length: MemoryLayout<PriorBoxMetalParam>.size, index: 1)
    
    encoder.setBytes(param.variances, length: MemoryLayout<Float32>.size * param.variances.count, index: 2)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
}
