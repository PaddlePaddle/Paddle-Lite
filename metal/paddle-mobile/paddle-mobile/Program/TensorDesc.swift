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
    let dataType: VarTypeType
    let dataLayout: DataLayout = DataLayout.NCHW()
    var NCHWDim: [Int] {
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
                fatalError(" not support other layout")
            }
        }
    }
    
    var NHWCDim: [Int] {
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
                fatalError(" not support other layout")
            }
        }
    }
    
    init(protoTensorDesc: PaddleMobile_Framework_Proto_VarType.TensorDesc) {
        dims = protoTensorDesc.dims.map{ Int($0) > 0 ? Int($0) : abs(Int($0)) }
        dataType = VarTypeType.init(rawValue: protoTensorDesc.dataType.rawValue) ?? .ErrorType
    }
    
}
