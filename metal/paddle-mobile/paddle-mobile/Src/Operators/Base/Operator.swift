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

import Metal
import Foundation

protocol Fusion {
    static func fusionNode() -> Node
    static func change() -> [String : [(from: String, to: String)]]
    static func fusionType() -> String
    static func needCheck() -> [(Int, String)]
}
extension Fusion {
    static func needCheck() -> [(Int, String)] {
        return []
    }
}

protocol Runable {
    func run(device: MTLDevice, buffer: MTLCommandBuffer) throws
    func runImpl(device: MTLDevice,buffer: MTLCommandBuffer) throws
    func delogOutput()
    func inputVariant() -> [String : [MTLBuffer]]?
    func computeMiddleResult(device: MTLDevice, buffer: MTLCommandBuffer) throws
}

extension Runable where Self: OperatorProtocol {
    func run(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        try runImpl(device: device, buffer: buffer)
    }
    
    func inputVariant() -> [String : [MTLBuffer]]? {
        //    return [:]
        return nil
    }
    
    func computeMiddleResult(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        throw PaddleMobileError.makeError(type: .netError, msg: "need implement func computeMiddleResult")
    }
    
    func delogOutput() {
        print(type + ": has no implementation" )
    }
}

public class InitContext {
    
    /// metal 代码加载方式
    var metalLoadMode: MetalLoadMode = .LoadMetalInDefaultLib
    
    /// 当 metalLoadMode 为 LoadMetalInCustomMetalLib 时， metal library 路径不能为空
    var metalLibPath: String? = nil
    
    /// 是否使用 MetalPerformanceShaders 进行运算
    var useMPS: Bool = false
    
    /// 是否使用最高等级的加速策略
    var useAggressiveOptimization: Bool = false
    
    init() {
        metalLoadMode = .LoadMetalInDefaultLib
        metalLibPath = nil
    }
}

protocol Creator where Self: OperatorProtocol{
    associatedtype OpType: OperatorProtocol & Runable & InferShaperable
    static func creat(device: MTLDevice, opDesc: PMOpDesc, inScope: Scope, initContext: InitContext) throws -> OpType
}

extension Creator where Self: OperatorProtocol {
    static func creat(device: MTLDevice, opDesc: PMOpDesc, inScope: Scope, initContext: InitContext) throws -> OpType {
        return try OpType.provide(device:device, opDesc: opDesc, inScope: inScope, initContext: initContext)
    }
}

protocol InferShaperable {
    func inferShape()
}

protocol OperatorProtocol {
    associatedtype ParamType
    associatedtype KerType:  Computable where Self.KerType.ParamType == ParamType
    var type: String { get }
    var scope: Scope { get }
    var inputs: [String : [String]] { get }
    var paraInputs: [String : [String]] { get set }
    var outpus: [String : [String]] { get }
    var attrs: [String : Attr] { get }
    var para: ParamType { get }
    var kernel: KerType { get }
    init(device: MTLDevice, opDesc: PMOpDesc, inScope: Scope, initContext: InitContext) throws
}

extension OperatorProtocol {
    static func provide(device: MTLDevice, opDesc: PMOpDesc, inScope: Scope, initContext: InitContext) throws -> Self {
        return try Self.init(device: device, opDesc: opDesc, inScope: inScope, initContext: initContext)
    }
}

class Operator <KernelType:  Computable , ParameterType>: OperatorProtocol where KernelType.ParamType == ParameterType {
    required init(device: MTLDevice, opDesc: PMOpDesc, inScope: Scope, initContext: InitContext) throws {
        type = opDesc.type
        scope = inScope
        inputs = opDesc.inputs
        outpus = opDesc.outputs
        attrs =  opDesc.attrs
        paraInputs = opDesc.paraInputs
        para = try ParamType.init(opDesc:opDesc, inScope: inScope)
        kernel = try KernelType.init(device: device, param: para, initContext: initContext)
    }
    
    typealias ParamType = ParameterType
    typealias KerType = KernelType
    let type: String
    let inputs: [String : [String]]
    var paraInputs: [String : [String]]
    let outpus: [String : [String]]
    let attrs: [String : Attr]
    let para: ParamType
    let scope: Scope
    var kernel: KerType
}

// op infos
let gFetchType                  = "fetch"
let gFeedType                   = "feed"
let gConvType                   = "conv2d"
let gBatchNormType              = "batch_norm"
let gReluType                   = "relu"
let gElementwiseAddType         = "elementwise_add"
let gConvAddBatchNormReluType   = "conv_add_batchnorm_relu"
let gPooType                    = "pool2d"
let gSoftmaxType                = "softmax"
let gReshapeType                = "reshape"
let gConvAddType                = "conv_add"
let gConvReluType               = "conv_relu"
let gDepthConvType              = "depthwise_conv2d"
let gPriorBoxType               = "prior_box"
let gTransposeType              = "transpose"
let gConcatType                 = "concat"
let gBoxcoderType               = "box_coder"
let gMulticlassNMSType          = "multiclass_nms"
let gConvBnReluType             = "conv_bn_relu"
let gDwConvBnReluType           = "depth_conv_bn_relu"
let gPreluType                  = "prelu"
let gConv2dTransposeType        = "conv2d_transpose"
let gBilinearInterpType         = "bilinear_interp"
let gSplit                      = "split"
let gShape                      = "shape"
let gFlatten                    = "flatten"
let gConvAddReluType            = "conv_add_relu"
let gConvAddPreluType           = "conv_add_prelu"
let gConvAddAddPreluType        = "conv_add_add_prelu"
let gElementwiseAddPreluType    = "elementwise_add_prelu"
let gFusionConvAddType          = "fusion_conv_add"
let gReshape2Type               = "reshape2"
let gTranspose2Type             = "transpose2"
let gScaleType                  = "scale"
let gRelu6Type                  = "relu6"
let gExpType                    = "exp"
let gSigmoidType                = "sigmoid"
let gLeakyReluType              = "leaky_relu"
let gFlatten2Type               = "flatten2"
let gSliceType                  = "slice"
let gNearestInterpType          = "nearest_interp"


let opInfos = [gConvType                    : (inputs: ["Input"], outputs: ["Output"]),
               gBatchNormType               : (inputs: ["X"], outputs: ["Y"]),
               gReluType                    : (inputs: ["X"], outputs: ["Out"]),
               gElementwiseAddType          : (inputs: ["X"], outputs: ["Out"]),
               gFeedType                    : (inputs: ["X"], outputs: ["Out"]),
               gFetchType                   : (inputs: ["X"], outputs: ["Out"]),
               gConvAddBatchNormReluType    : (inputs: ["Input"], outputs: ["Out"]),
               gPooType                     : (inputs: ["X"], outputs: ["Out"]),
               gSoftmaxType                 : (inputs: ["X"], outputs: ["Out"]),
               gReshapeType                 : (inputs: ["X"], outputs: ["Out"]),
               gConvAddType                 : (inputs: ["Input"], outputs: ["Out"]),
               gDepthConvType               : (inputs: ["Input"], outputs: ["Output"]),
               gConcatType                  : (inputs: ["X"], outputs: ["Out"]),
               gBoxcoderType                : (inputs: ["PriorBox", "PriorBoxVar", "TargetBox"], outputs: ["OutputBox"]),
               gTransposeType               : (inputs: ["X"], outputs: ["Out"]),
               gConvBnReluType              : (inputs: ["Input"], outputs: ["Out"]),
               gDwConvBnReluType            : (inputs: ["Input"], outputs: ["Out"]),
               gMulticlassNMSType           : (inputs: ["BBoxes", "Scores"], outputs: ["Out"]),
               gPriorBoxType                : (inputs: ["Input", "Image"], outputs: ["Boxes", "Variances"]),
               gPreluType                   : (inputs: ["X"], outputs: ["Out"]),
               gConv2dTransposeType         : (inputs: ["Input"], outputs: ["Output"]),
               gBilinearInterpType          : (inputs: ["X"], outputs: ["Out"]),
               gSplit                       : (inputs: ["X"], outputs: ["Out"]),
               gShape                       : (inputs: ["Input"], outputs: ["Out"]),
               gFlatten                     : (inputs: ["X"], outputs: ["Out"]),
               gConvAddPreluType            : (inputs: ["Input"], outputs: ["Out"]),
               gConvAddAddPreluType         : (inputs: ["Input"], outputs: ["Out"]),
               gElementwiseAddPreluType     : (inputs: ["X"], outputs: ["Out"]),
               gFusionConvAddType           : (inputs: ["Input"], outputs: ["Out"]),
               gReshape2Type                : (inputs: ["X"], outputs: ["Out"]),
               gTranspose2Type              : (inputs: ["X"], outputs: ["Out"]),
               gScaleType                   : (inputs: ["X"], outputs: ["Out"]),
               gRelu6Type                   : (inputs: ["X"], outputs: ["Out"]),
               gExpType                     : (inputs: ["X"], outputs: ["Out"]),
               gSigmoidType                 : (inputs: ["X"], outputs: ["Out"]),
               gLeakyReluType               : (inputs: ["X"], outputs: ["Out"]),
               gFlatten2Type                : (inputs: ["X"], outputs: ["Out"]),
               gSliceType                   : (inputs: ["Input"], outputs: ["Out"]),
               gNearestInterpType           : (inputs: ["X"], outputs: ["Out"]),
]
