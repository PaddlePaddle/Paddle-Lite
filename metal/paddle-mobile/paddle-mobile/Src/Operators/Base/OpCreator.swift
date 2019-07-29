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
class OpCreator<P: PrecisionProtocol> {
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
    
    func creat(device: MTLDevice, opDesc: PMOpDesc, scope: Scope, initContext: InitContext) throws -> Runable & InferShaperable {
        guard let opCreator = opCreators[opDesc.type] else {
            throw PaddleMobileError.makeError(type: .opError, msg: "there is no " + opDesc.type + " yet")
        }
        
        return try opCreator(device, opDesc, scope, initContext)
    }
    
    let opCreators: [String : (MTLDevice, PMOpDesc, Scope, InitContext) throws -> Runable & InferShaperable] =
        [gConvType                  :     ConvOp<P>.creat,
         gBatchNormType             :     BatchNormOp<P>.creat,
         gReluType                  :     ReluOp<P>.creat,
         gElementwiseAddType        :     ElementwiseAddOp<P>.creat,
         gFeedType                  :     FeedOp<P>.creat,
         gFetchType                 :     FetchOp<P>.creat,
         gConvAddBatchNormReluType  :     ConvAddBatchNormReluOp<P>.creat,
         gPooType                   :     PoolOp<P>.creat,
         gSoftmaxType               :     SoftmaxOp<P>.creat,
         gReshapeType               :     ReshapeOp<P>.creat,
         gConvAddType               :     ConvAddOp<P>.creat,
         gDepthConvType             :     DepthConvOp<P>.creat,
         gConcatType                :     ConcatOp<P>.creat,
         gBoxcoderType              :     BoxcoderOp<P>.creat,
         gConvBnReluType            :     ConvBNReluOp<P>.creat,
         gDwConvBnReluType          :     DwConvBNReluOp<P>.creat,
         gMulticlassNMSType         :     MulticlassNMSOp<P>.creat,
         gTransposeType             :     TransposeOp<P>.creat,
         gPriorBoxType              :     PriorBoxOp<P>.creat,
         gPreluType                 :     PreluOp<P>.creat,
         gConv2dTransposeType       :     ConvTransposeOp<P>.creat,
         gBilinearInterpType        :     BilinearInterpOp<P>.creat,
         gSplit                     :     SplitOp<P>.creat,
         gShape                     :     ShapeOp<P>.creat,
         gFlatten                   :     FlattenOp<P>.creat,
         gConvAddPreluType          :     ConvAddPreluOp<P>.creat,
         gConvAddAddPreluType       :     ConvAddAddPreluOp<P>.creat,
         gElementwiseAddPreluType   :     ElementwiseAddPreluOp<P>.creat,
         gFusionConvAddType         :     ConvAddOp<P>.creat,
         gConvAddReluType           :     ConvAddReluOp<P>.creat,
         gReshape2Type              :     ReshapeOp<P>.creat,
         gTranspose2Type            :     TransposeOp<P>.creat,
         gScaleType                 :     ScaleOp<P>.creat,
         gRelu6Type                 :     Relu6Op<P>.creat,
         gExpType                   :     ExpOp<P>.creat,
         gSigmoidType               :     SigmoidOp<P>.creat,
         gLeakyReluType             :     LeakyReluOp<P>.creat,
         gFlatten2Type              :     Flatten2Op<P>.creat,
         gSliceType                 :     SliceOp<P>.creat,
         gNearestInterpType         :     NearestInterpOp<P>.creat,
         gConvReluType              :     ConvReluOp<P>.creat,
    ]
    
    
    private init(){}
}
