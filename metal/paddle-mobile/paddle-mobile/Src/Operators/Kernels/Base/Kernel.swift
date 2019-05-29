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
    init(device: MTLDevice, testParam: TestParamType, initContext: InitContext)
}


protocol Computable {
    associatedtype ParamType: OpParam
    func compute(commandBuffer: MTLCommandBuffer, param: ParamType) throws
    init(device: MTLDevice, param: ParamType, initContext: InitContext) throws
}

protocol KernelProtocol {
    var pipline: MTLComputePipelineState { get set }
    var functionName: String { get set }
    
}

@objc open class Kernel: NSObject {
    
    private var _pipline: MTLComputePipelineState? = nil
    
    var pipline: MTLComputePipelineState {
        get {
            return _pipline ?! " pipeline can't be nil "
        }
    }
    
    let functionName: String?
    public init(device: MTLDevice, inFunctionName: String?, usePaddleMobileLib: Bool = false, initContext: InitContext) {
        functionName = inFunctionName
        if let funcName = inFunctionName {
            _pipline = device.pipeLine(funcName: funcName, metalLoadMode: initContext.metalLoadMode, metalLibPath: initContext.metalLibPath)
        }
    }
    
    func encodeTransposeInput(input: Texture, toTranspose: [Int], commandBuffer: MTLCommandBuffer, device: MTLDevice, initContext: InitContext) -> Texture? {
        do {
            let intermediateTexture = Texture(device: device, inDim: input.tensorDim)
            try intermediateTexture.initTexture(device: device, inTranspose: toTranspose, computePrecision: GlobalConfig.shared.computePrecision)
            
            let irank = input.tensorDim.cout()
            let orank = intermediateTexture.tensorDim.cout()
            var funcName = ""
            if GlobalConfig.shared.computePrecision == .Float32 {
                funcName = "reshape_\(irank)_\(orank)_float"
            } else if GlobalConfig.shared.computePrecision == .Float16 {
                funcName = "reshape_\(irank)_\(orank)_half"
            } else {
                fatalError()
            }
            let intermediatePipeline = device.pipeLine(funcName: funcName, metalLoadMode: initContext.metalLoadMode, metalLibPath: initContext.metalLibPath)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PaddleMobileError.predictError(message: " encode is nil")
            }
            encoder.setTexture(input.metalTexture, index: 0)
            encoder.setTexture(intermediateTexture.metalTexture, index: 1)
            var id: [Int32] = [1, 1, 1, 1]
            for i in 0..<input.tensorDim.cout() {
                id[4-input.tensorDim.cout()+i] = Int32(input.tensorDim[i])
            }
            let it: [Int32] = input.transpose.map { Int32($0) }
            var od: [Int32] = [1, 1, 1, 1]
            for i in 0..<intermediateTexture.tensorDim.cout() {
                od[4-intermediateTexture.tensorDim.cout()+i] = Int32(intermediateTexture.tensorDim[i])
            }
            let ot: [Int32] = intermediateTexture.transpose.map { Int32($0) }
            var reshapeMetalParam = ReshapeMetalParam.init(
                idim: (id[0], id[1], id[2], id[3]),
                itrans: (it[0], it[1], it[2], it[3]),
                odim: (od[0], od[1], od[2], od[3]),
                otrans: (ot[0], ot[1], ot[2], ot[3])
            )
            encoder.setBytes(&reshapeMetalParam, length: MemoryLayout<ReshapeMetalParam>.size, index: 0)
            encoder.dispatch(computePipline: intermediatePipeline, outTexture: intermediateTexture.metalTexture)
            encoder.endEncoding()
            return intermediateTexture
        } catch _ {
            return nil
        }
    }
}

@objc public class Shape: NSObject {
    public let width: Int
    public let height: Int
    public let channel: Int
    @objc public init(inWidth: Int, inHeight: Int, inChannel: Int){
        width = inWidth
        height = inHeight
        channel = inChannel
    }
}

open class BufferToTextureKernel: Kernel {
    public let outputTexture: MTLTexture
    
    public init(device: MTLDevice, outputDim: Shape, metalLoadMode: MetalLoadMode, metalLibPath: String?) {
        let textureDesc = MTLTextureDescriptor.init()
        textureDesc.textureType = .type2D
        textureDesc.width = outputDim.width
        textureDesc.height = outputDim.height
        textureDesc.depth = (outputDim.channel + 3) / 4
        
        if GlobalConfig.shared.computePrecision == .Float16 {
            textureDesc.pixelFormat = .rgba16Float
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            textureDesc.pixelFormat = .rgba32Float
        } else {
            fatalError()
        }
        
        textureDesc.usage = [.shaderRead, .shaderWrite]
        textureDesc.storageMode = .shared
        outputTexture = device.makeTexture(descriptor: textureDesc) ?! " make texture error "
        let initContext = InitContext.init()
        initContext.metalLibPath = metalLibPath
        initContext.metalLoadMode = metalLoadMode
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "buffer_to_texture_kernel", initContext: initContext)
        } else {
            super.init(device: device, inFunctionName: "buffer_to_texture_kernel_half", initContext: initContext)
        }
    }
    
    public func compute(inputBuffer: MTLBuffer , commandBuffer: MTLCommandBuffer) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setTexture(outputTexture, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: outputTexture)
        encoder.endEncoding()
    }
    
}

@objc open class CusomKernel: Kernel {
    
    public let outputTexture: MTLTexture
    public init(device: MTLDevice, inFunctionName: String?, outputDim: Shape, metalLoadModel: MetalLoadMode, metalLibPath: String?) {
        let textureDesc = MTLTextureDescriptor.init()
        textureDesc.textureType = .type2D
        textureDesc.width = outputDim.width
        textureDesc.height = outputDim.height
        textureDesc.depth = (outputDim.channel + 3) / 4
        
        if GlobalConfig.shared.computePrecision == .Float16 {
            textureDesc.pixelFormat = .rgba16Float
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            textureDesc.pixelFormat = .rgba32Float
        } else {
            fatalError()
        }
        
        textureDesc.usage = [.shaderRead, .shaderWrite]
        textureDesc.storageMode = .shared
        outputTexture = device.makeTexture(descriptor: textureDesc) ?! " make texture error "
        
        let context = InitContext.init()
        context.metalLoadMode = metalLoadModel
        context.metalLibPath = metalLibPath
        super.init(device: device, inFunctionName: inFunctionName, initContext: context)
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

