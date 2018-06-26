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

//typealias Float16 = Int16
//extension Float16: PrecisionType {
//}

public protocol PrecisionType {
}

extension Float32: PrecisionType {
}

enum DataLayout {
    case NCHW
    case NHWC
}

protocol Variant {
}

extension Tensor: Variant {
    
}
extension Texture: Variant {
}


let gFetchType          = "fetch"
let gFeedType           = "feed"
let gConvType           = "conv2d"
let gBatchNormType      = "batch_norm"
let gReluType           = "relu"
let gElementwiseAdd     = "elementwise_add"


fileprivate var singletons : [String : Any] = [:]
class OpCreator<P: PrecisionType> {
    static var shared : OpCreator<P> {
        let key = String(describing: P.self)
        if let singleton = singletons[key] {
            return singleton as! OpCreator<P>
        } else {
            let newSingleton = OpCreator<P>()
            singletons[key] = newSingleton
            return newSingleton
        }
    }
    
    func creat(opDesc: OpDesc, scope: Scope) throws -> Runable {
        guard let opCreator = opCreators[opDesc.type] else {
            throw PaddleMobileError.opError(message: "there is no " + opDesc.type + " yet")
        }
        
        do {
            return try opCreator(opDesc, scope)
        } catch let error {
            throw error
        }
    }
    
    let opCreators: [String : (OpDesc, Scope) throws -> Runable] =
                    [gConvType        :     ConvOp<P>.creat,
                    gBatchNormType    :     BatchNormOp<P>.creat,
                    gReluType         :     ReluOp<P>.creat,
                    gElementwiseAdd   :     ElementwiseAddOp<P>.creat,
                    gFeedType         :     FeedOp<P>.creat,
                    gFetchType        :     FetchOp<P>.creat]
    
    private init(){}
}

let opInfos = [gConvType         : (inputs: ["Input"], outputs: ["Output"]),
               gBatchNormType    : (inputs: ["X"], outputs: ["Y"]),
               gReluType         : (inputs: ["X"], outputs: ["Out"]),
               gElementwiseAdd   : (inputs: ["X", "Y"], outputs: ["Out"]),
               gFeedType         : (inputs: ["X"], outputs: ["Out"]),
               gFetchType        : (inputs: ["X"], outputs: ["Out"])]

