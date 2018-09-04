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

struct TransposeMetalParam {
  var iC: Int32 = 0
  var oC: Int32 = 0
  var i0: Int32
  var i1: Int32
  var i2: Int32
  var i3: Int32
  init(_ i0: Int32, _ i1: Int32, _ i2: Int32, _ i3: Int32) {
    self.i0 = i0
    self.i1 = i1
    self.i2 = i2
    self.i3 = i3
  }
  init(_ axis: [Int]) {
    self.init(Int32(axis[0]), Int32(axis[1]), Int32(axis[2]), Int32(axis[3]))
  }
}

struct TransposeTestParam: TestParam {
  let inputTexture: MTLTexture
  let outputTexture: MTLTexture
  let iC: Int
  let oC: Int
  let axis: [Int]
}

class TransposeKernel<P: PrecisionType>: Kernel, Computable, Testable {
  
  required init(device: MTLDevice, param: TransposeParam<P>) {
    param.output.initTexture(device: device, inTranspose: [0, 1, 2, 3], computePrecision: computePrecision)
    
    if computePrecision == .Float16 {
      super.init(device: device, inFunctionName: "transpose_half")
    } else if computePrecision == .Float32 {
      super.init(device: device, inFunctionName: "transpose")
    } else {
      fatalError()
    }
    var invT: [Int] = [0, 1, 2, 3]
    for (i, v) in param.input.transpose.enumerated() {
      invT[v] = i
    }
    var axis: [Int] = [0, 1, 2, 3]
      
    for i in 0..<param.axis.count {
      axis[4-param.axis.count+i] = 4 - param.axis.count + Int(param.axis[i])
    }
    let realAxis = axis.map {invT[$0]}
    var tmp = TransposeMetalParam.init(realAxis)
    tmp.iC = Int32(param.input.dim[param.input.transpose[3]])
    tmp.oC = Int32(param.output.dim[3])
    if realAxis == [0, 1, 2, 3] {
//      print("====> transpose! FAST :)")
    } else {
//      print("====> transpose! SLOW :(")
    }
    metalParam = tmp
  }
  
  required init(device: MTLDevice, testParam: TransposeTestParam) {
    if computePrecision == .Float16 {
      super.init(device: device, inFunctionName: "transpose_half")
    } else if computePrecision == .Float32 {
      super.init(device: device, inFunctionName: "transpose")
    } else {
      fatalError()
    }
  }
  
  var metalParam: TransposeMetalParam!
  func compute(commandBuffer: MTLCommandBuffer, param: TransposeParam<P>) throws {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw PaddleMobileError.predictError(message: " encode is nil")
    }
  
    encoder.setTexture(param.input.metalTexture, index: 0)
    encoder.setTexture(param.output.metalTexture, index: 1)
    encoder.setBytes(&metalParam, length: MemoryLayout<TransposeMetalParam>.size, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
    encoder.endEncoding()
  }

  
  public func test(commandBuffer: MTLCommandBuffer, param: TransposeTestParam) {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError()
    }
    
    encoder.setTexture(param.inputTexture, index: 0)
    encoder.setTexture(param.outputTexture, index: 1)
    var tmp = TransposeMetalParam.init(param.axis)
    tmp.iC = Int32(param.iC)
    tmp.oC = Int32(param.oC)
    
    encoder.setBytes(&tmp, length: MemoryLayout<TransposeMetalParam>.size, index: 0)
    encoder.dispatch(computePipline: pipline, outTexture: param.outputTexture)
    encoder.endEncoding()
  }}
