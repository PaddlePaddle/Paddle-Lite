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
}

protocol Runable {
    func run(device: MTLDevice, buffer: MTLCommandBuffer) throws
    func runImpl(device: MTLDevice,buffer: MTLCommandBuffer) throws
    func delogOutput()
}

extension Runable where Self: OperatorProtocol{
    func run(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        do {
            try runImpl(device: device, buffer: buffer)
        } catch let error {
            throw error
        }
//        print(type + ": " + para.outputDesc())
    }
    
    func delogOutput() {
        print(type + ": has no implementation" )
    }
}

protocol Creator where Self: OperatorProtocol{
    associatedtype OpType: OperatorProtocol & Runable & InferShaperable
    static func creat(device: MTLDevice, opDesc: OpDesc, inScope: Scope) throws -> OpType
}

extension Creator where Self: OperatorProtocol {
    static func creat(device: MTLDevice, opDesc: OpDesc, inScope: Scope) throws -> OpType {
        do {
            return try OpType.provide(device:device, opDesc: opDesc, inScope: inScope)
        } catch let error {
            throw error
        }
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
    init(device: MTLDevice, opDesc: OpDesc, inScope: Scope) throws
}

extension OperatorProtocol {
    static func provide(device: MTLDevice, opDesc: OpDesc, inScope: Scope) throws -> Self {
        do {
            return try Self.init(device: device, opDesc: opDesc, inScope: inScope)
        } catch let error {
            throw error
        }
    }
}

class Operator <KernelType:  Computable , ParameterType>: OperatorProtocol where KernelType.ParamType == ParameterType {
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
    required init(device: MTLDevice, opDesc: OpDesc, inScope: Scope) throws {
        type = opDesc.type
        scope = inScope
        inputs = opDesc.inputs
        outpus = opDesc.outputs
        attrs =  opDesc.attrs
        paraInputs = opDesc.paraInputs
        do {
            para = try ParamType.init(opDesc:opDesc, inScope: inScope)
        } catch let error {
            throw error
        }
        kernel = KernelType.init(device: device, param: para)
    }
}

// op infos
let gFetchType                  = "fetch"
let gFeedType                   = "feed"
let gConvType                   = "conv2d"
let gBatchNormType              = "batch_norm"
let gReluType                   = "relu"
let gElementwiseAdd             = "elementwise_add"
let gConvAddBatchNormReluType   = "conv_add_batchnorm_relu"
let gPooType                    = "pool2d"
let gSoftmaxType                = "softmax"
let gReshapeType                = "reshape"
let gConvAddType                = "conv_add"
let gMPSCNNConvType             = "mps_cnn_conv"
let gBatchNormReluType          = "batch_norm_relu"


let opInfos = [gConvType                    : (inputs: ["Input"], outputs: ["Output"]),
               gBatchNormType               : (inputs: ["X"], outputs: ["Y"]),
               gReluType                    : (inputs: ["X"], outputs: ["Out"]),
               gElementwiseAdd              : (inputs: ["X"], outputs: ["Out"]),
               gFeedType                    : (inputs: ["X"], outputs: ["Out"]),
               gFetchType                   : (inputs: ["X"], outputs: ["Out"]),
               gConvAddBatchNormReluType    : (inputs: ["Input"], outputs: ["Out"]),
               gPooType                     : (inputs: ["X"], outputs: ["Out"]),
               gSoftmaxType                 : (inputs: ["X"], outputs: ["Out"]),
               gReshapeType                 : (inputs: ["X"], outputs: ["Out"]),
               gConvAddType                 : (inputs: ["Input"], outputs: ["Out"]),
               gMPSCNNConvType              : (inputs: ["Input"], outputs: ["Out"]),
               gBatchNormReluType           : (inputs: ["X"], outputs: ["Out"])]
