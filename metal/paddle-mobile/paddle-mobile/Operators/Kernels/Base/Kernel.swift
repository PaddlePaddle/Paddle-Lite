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

public protocol TestParam {
}

public protocol Testable {
    associatedtype TestParamType: TestParam
    func test(commandBuffer: MTLCommandBuffer, param: TestParamType)
    init(device: MTLDevice, testParam: TestParamType)
}


protocol Computable {
    associatedtype ParamType: OpParam
    func compute(commandBuffer: MTLCommandBuffer, param: ParamType) throws
    init(device: MTLDevice, param: ParamType)
}

protocol KernelProtocol {
    var pipline: MTLComputePipelineState { get set }
    var functionName: String { get set }
   
}

open class Kernel {
    let pipline: MTLComputePipelineState
    let functionName: String
    public init(device: MTLDevice, inFunctionName: String, usePaddleMobileLib: Bool = true) {
        pipline = device.pipeLine(funcName: inFunctionName, inPaddleMobileLib: usePaddleMobileLib)
        functionName = inFunctionName
    }
}

open class CusomKernel: Kernel {
    public struct Shape {
        public let width: Int
        public let height: Int
        public let channel: Int
        public init(inWidth: Int, inHeight: Int, inChannel: Int){
            width = inWidth
            height = inHeight
            channel = inChannel
        }
    }
    public let outputTexture: MTLTexture
    public init(device: MTLDevice, inFunctionName: String, outputDim: Shape, usePaddleMobileLib: Bool = false) {
        let textureDesc = MTLTextureDescriptor.init()
        textureDesc.textureType = .type2D
        textureDesc.width = outputDim.width
        textureDesc.height = outputDim.height
        textureDesc.depth = (outputDim.channel + 3) / 4
        textureDesc.pixelFormat = .rgba32Float
        textureDesc.usage = [.shaderRead, .shaderWrite]
        textureDesc.storageMode = .shared
        outputTexture = device.makeTexture(descriptor: textureDesc) ?! " make texture error "

        super.init(device: device, inFunctionName: inFunctionName, usePaddleMobileLib: usePaddleMobileLib)
    }
    
    public func compute(inputTexuture: MTLTexture, commandBuffer: MTLCommandBuffer) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        encoder.setTexture(inputTexuture, index: 0)
        encoder.setTexture(outputTexture, index: 1)
        encoder.dispatch(computePipline: pipline, outTexture: outputTexture)
        encoder.endEncoding()
    }
    
}

