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
import Metal

public class ResultHolder: NSObject {
  @objc public let result: UnsafeMutablePointer<Float32>?
  @objc public let capacity: Int

  init(inResult: UnsafeMutablePointer<Float32>?, inCapacity: Int) {
    result = inResult
    capacity = inCapacity
  }
  
  @objc public func releasePointer() {
    result?.deinitialize(count: capacity)
    result?.deallocate()
  }
}

public class Net: NSObject {
  var except: Int = 0
  
  // for CPU
  var means: [Float] = []
  var scale: Float = 0.0
  
  var needUpdateProgram = true
  
  public var inputDim: Dim {
    get{
      return inputDim_
    }
    set{
      if inputDim_ != newValue {
        needUpdateProgram = true
      }
      inputDim_ = newValue
    }
  }
  
  var inputDim_: Dim = Dim.init(inDim: [])
  var preprocessKernel: CusomKernel? = nil
  var paramPointer: UnsafeMutableRawPointer? = nil
  var paramSize: Int = 0
  var modelPointer: UnsafeMutableRawPointer? = nil
  var modelSize: Int = 0
  var modelPath: String = ""
  var paramPath: String = ""
  var modelDir: String = ""
  let device: MTLDevice
  
  @objc public init(device: MTLDevice,paramPointer: UnsafeMutableRawPointer, paramSize:Int, modePointer: UnsafeMutableRawPointer, modelSize: Int) {
    self.paramPointer = paramPointer
    self.paramSize = paramSize
    self.modelPointer = modePointer
    self.modelSize = modelSize
    self.device = device
    super.init()
  }
  
  @objc public init(device: MTLDevice) {
    self.device = device
    super.init()
  }

  @objc public func updateInputDim(inDim: [Int]) {
    inputDim = Dim.init(inDim: inDim)
  }
  
  public func resultStr(res: ResultHolder) -> String {
    fatalError()
  }
  
  func fetchResult(paddleMobileRes: GPUResultHolder) -> ResultHolder {
    return ResultHolder.init(inResult: paddleMobileRes.resultPointer, inCapacity: paddleMobileRes.capacity)
  }
  
  func updateProgram(program: Program) {
  }
}
