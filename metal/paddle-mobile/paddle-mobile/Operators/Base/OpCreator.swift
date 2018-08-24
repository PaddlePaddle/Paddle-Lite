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

fileprivate var singletons : [String : Any] = [:]
@available(iOS 10.0, *)
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
    
    func creat(device: MTLDevice, opDesc: OpDesc, scope: Scope) throws -> Runable & InferShaperable {
        guard let opCreator = opCreators[opDesc.type] else {
            throw PaddleMobileError.opError(message: "there is no " + opDesc.type + " yet")
        }
        
        do {
            return try opCreator(device, opDesc, scope)
        } catch let error {
            throw error
        }
    }
    
    let opCreators: [String : (MTLDevice, OpDesc, Scope) throws -> Runable & InferShaperable] =
        [gConvType                  :     ConvOp<P>.creat,
         gBatchNormType             :     BatchNormOp<P>.creat,
         gReluType                  :     ReluOp<P>.creat,
         gElementwiseAdd            :     ElementwiseAddOp<P>.creat,
         gFeedType                  :     FeedOp<P>.creat,
         gFetchType                 :     FetchOp<P>.creat,
         gConvAddBatchNormReluType  :     ConvAddBatchNormReluOp<P>.creat,
         gPooType                   :     PoolOp<P>.creat,
         gSoftmaxType               :     SoftmaxOp<P>.creat,
         gReshapeType               :     ReshapeOp<P>.creat,
         gConvAddType               :     ConvAddOp<P>.creat,
         gMPSCNNConvType            :     CNNConvAddBatchNormReluOp<P>.creat,
         gBatchNormReluType         :     BatchNormReluOp<P>.creat]
    
    private init(){}
}
