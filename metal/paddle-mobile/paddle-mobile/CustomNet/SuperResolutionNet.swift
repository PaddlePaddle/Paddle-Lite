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

public class SuperResolutionNet: Net{
  override public func resultStr(res: ResultHolder) -> String {
    return "未实现"
  }
  
  override public init(device: MTLDevice) {
    super.init(device: device)
    means = [0.0, 0.0, 0.0]
    scale = 1.0
    except = 0
    modelPath = Bundle.main.path(forResource: "super_model", ofType: nil) ?! "model null"
    paramPath = Bundle.main.path(forResource: "super_params", ofType: nil) ?! "para null"
    modelDir = ""
    preprocessKernel = nil
//    inputDim_ = Dim.init(inDim: [1, Int(552 * 1.414), Int(310 * 1.414), 1])
    inputDim_ = Dim.init(inDim: [1, 224, 224, 1])
  }
  
  override func updateProgram(program: Program) {
    guard needUpdateProgram else {
      return
    }
    
    // n h w c
    for block in program.programDesc.blocks {
      for varDesc in block.vars {
        if !varDesc.persistable {
          if varDesc.type == .LodTensor {
            let varEle = program.scope.vars[varDesc.name]
            if let texture = varEle as? Texture {
              let newDim = Dim.init(inDim: [texture.dim[0],  inputDim[1], inputDim[2], texture.tensorDim[1]])
              print(" var desc name " + varDesc.name + " new dim" + "\(newDim)")
              texture.updateDims(inTensorDim: Dim.init(inDim: [texture.tensorDim[0], texture.tensorDim[1], inputDim[1], inputDim[2]]), inDim: newDim)
              texture.initTexture(device: device, inTranspose: [0, 1, 2, 3], computePrecision: computePrecision)
              
              let output: FetchHolder = program.scope.output() as! FetchHolder
              output.dim = newDim
              output.capacity = newDim.numel()
              output.paddedCapacity = newDim.numel() * 4
              output.initBuffer(device: device)
            }
          }
        }
      }
    }
    needUpdateProgram = false
  }
}

