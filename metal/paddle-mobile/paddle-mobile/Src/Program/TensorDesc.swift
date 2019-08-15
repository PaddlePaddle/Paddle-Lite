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

class TensorDesc {
    let dims: [Int]
    let originDimsCount: Int
    let dataType: VarTypeType
    let dataLayout: DataLayout = DataLayout.NCHW()
    var NCHWDim: [Int]? {
        get {
            if dims.count != 4 {
                return dims
            }
            if dataLayout == DataLayout.NCHW() {
                return dims
            } else if dataLayout == DataLayout.NHWC() {
                var resultDims = dims
                resultDims.swapAt(1, 3)
                return resultDims
            } else {
                paddleMobileLog("not support other layout", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                return nil
            }
        }
    }
    
    var NHWCDim: [Int]? {
        get {
            if dims.count != 4 {
                return dims
            }
            if dataLayout == DataLayout.NHWC() {
                return dims
            } else if dataLayout == DataLayout.NCHW() {
                var resultDims = dims
                resultDims.swapAt(1, 3)
                return resultDims
            } else {
                paddleMobileLog("not support other layout", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                return nil
            }
        }
    }
    
    init?(protoTensorDesc: VarType_TensorDesc) {
        //        dims = protoTensorDesc.dimsArray.map{ Int64($0)! > 0 ? Int64($0) : abs(Int64($0)) }
        
        var dimsArray = [Int]()
        
        let dimsCount = protoTensorDesc.dimsArray.count
        for i in 0..<dimsCount {
            let dim = Int(protoTensorDesc.dimsArray.value(at: i)) > 0 ?Int(protoTensorDesc.dimsArray.value(at: i)) :abs(Int(protoTensorDesc.dimsArray.value(at: i)))
            dimsArray.append(dim)
        }
        
        originDimsCount = Int(dimsCount)
        
        if dimsCount > 4 {
            let headDims = Int(dimsCount - 4)
            for i in 0..<headDims {
                guard dimsArray[i] <= 1 else {
                    paddleMobileLog("dims count is larger than 4 and can't be truncated to 4", logLevel: .FatalError, callStack: Thread.callStackSymbols)
                    return nil
                }
            }
            for _ in 0..<headDims {
                dimsArray.removeFirst()
            }
        }
        
        dims = dimsArray
        
        dataType = VarTypeType.init(rawValue: Int(protoTensorDesc.dataType.rawValue)) ?? .ErrorType
    }
    
}
