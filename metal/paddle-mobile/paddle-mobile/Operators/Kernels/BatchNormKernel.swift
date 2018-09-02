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

class BatchNormKernel<P: PrecisionType>: Kernel, Computable {
  var newScale: MTLBuffer
  var newBias: MTLBuffer
  
  required init(device: MTLDevice, param: BatchNormParam<P>) {
    guard let newScale = device.makeBuffer(length: param.inputScale.buffer.length) else {
      fatalError()
    }
    
    guard let newBias = device.makeBuffer(length: param.inputBias.buffer.length) else {
      fatalError()
    }
    self.newScale = newScale
    self.newBias = newBias
    
    if computePrecision == .Float32 {
      super.init(device: device, inFunctionName: "batchnorm")
    } else if computePrecision == .Float16 {
      super.init(device: device, inFunctionName: "batchnorm_half")
    } else {
      fatalError()
    }
    
    let varianceBuffer : MTLBuffer = param.inputVariance.buffer
    
    var invStd: [Float32] = Array(repeating: 0, count: varianceBuffer.length)
    let varianceContents = varianceBuffer.contents().assumingMemoryBound(to: P.self)
    for i in 0..<(varianceBuffer.length / MemoryLayout<P>.stride) {
      invStd[i] = 1 / (Float32(varianceContents[i]) + param.epsilon).squareRoot()
    }
    
    let newScaleContents = newScale.contents().assumingMemoryBound(to: P.self)
    let newBiasContents = newBias.contents().assumingMemoryBound(to: P.self)
    let scale : MTLBuffer = param.inputScale.buffer
    let scaleContents = scale.contents().assumingMemoryBound(to: P.self)
    let bias : MTLBuffer = param.inputBias.buffer
    let biasContents = bias.contents().assumingMemoryBound(to: P.self)
    let meanContents = param.inputMean.buffer.contents().assumingMemoryBound(to: P.self)
    
    for i in 0..<(newScale.length / MemoryLayout<P>.stride) {
      newScaleContents[i] = P(invStd[i] * Float32(scaleContents[i]))
      newBiasContents[i] = P(Float32(biasContents[i]) - Float32(meanContents[i]) * invStd[i] * Float32(scaleContents[i]))
    }
  }
  
  func compute(commandBuffer: MTLCommandBuffer, param: BatchNormParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encoder is nil")
    }
    print("BatchNorm compute")
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setTexture(param.output.metalTexture, index: 1)
    encoder.setBuffer(newScale, offset: 0, index: 0)
    encoder.setBuffer(newBias, offset: 0, index: 1)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }
}
