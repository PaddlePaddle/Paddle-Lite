/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/
import Foundation
import MetalPerformanceShaders

class FuncConstantBase {
    let type: MTLDataType
    let index: Int
    
    init(type: MTLDataType, index: Int) {
        self.index = index
        self.type = type
    }
    
    func getValue() -> Any {
        fatalError()
    }
}

class  FuncConstant<T: Any>: FuncConstantBase, Equatable where T: Equatable{
    let value: T
    init(index: Int, type: MTLDataType, value: T) {
        self.value = value
        super.init(type: type, index: index)
    }
    
    override func getValue() -> Any {
        return value
    }
    
    static func ==(lhs: FuncConstant, rhs: FuncConstant) -> Bool {
        return lhs.index == rhs.index && lhs.type == rhs.type && lhs.value == rhs.value
    }
}

@available(iOS 10.0, *)
final class MetalManager {
    static let shared: MetalManager = MetalManager()
    let device: MTLDevice
    let mdlLibrary: MTLLibrary
    let mainLibrary: MTLLibrary?
    private init(){
        device = MTLCreateSystemDefaultDevice()!
        mdlLibrary = device.makeLibrary(bundle: Bundle(for: MetalManager.self)) ?! ""
        mainLibrary = device.makeLibrary(bundle: Bundle.main)
    }
    
    static func makeFunction(device: MTLDevice, name: String, bundle: Bundle = Bundle(for: MetalManager.self), constantValues: MTLFunctionConstantValues? = nil) -> MTLComputePipelineState {
        if bundle == Bundle.main {
            return makeFunction(library: MetalManager.shared.mainLibrary ?! "no metal in this bundle", name: name, constantValues: constantValues)
        }
        return makeFunction(library: MetalManager.shared.mdlLibrary, name: name, constantValues: constantValues)
    }
    
    static func makeFunction(library: MTLLibrary, name: String,
                             constantValues: MTLFunctionConstantValues? = nil) -> MTLComputePipelineState {
        do {
            if let constantValues = constantValues {
                let kernelFunction = try library.makeFunction(name: name, constantValues: constantValues)
                return try library.device.makeComputePipelineState(function: kernelFunction)
            } else {
                guard let kernelFunction = library.makeFunction(name: name) else {
                    fatalError("Could not load compute function '\(name)'")
                }
                return try library.device.makeComputePipelineState(function: kernelFunction)
            }
        } catch {
            fatalError("Could not create compute pipeline for function '\(name)'")
        }
    }
    
    static func makeBuffer(device: MTLDevice,
                           channelFormat: MPSImageFeatureChannelFormat,
                           kernelWidth: Int,
                           kernelHeight: Int,
                           inputFeatureChannels: Int,
                           outputFeatureChannels: Int,
                           weights: UnsafePointer<Float>) -> MTLBuffer {
        
        assert(channelFormat == .float16)
        
        let inputSlices = (inputFeatureChannels + 3) / 4
        let paddedInputChannels = inputSlices * 4
        let outputSlices = (outputFeatureChannels + 3) / 4
        let paddedOutputChannels = outputSlices * 4
        let count = paddedOutputChannels * kernelHeight * kernelWidth * paddedInputChannels
        
        let buffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)
        
        copy(weights: weights, to: buffer, channelFormat: channelFormat,
             kernelWidth: kernelWidth, kernelHeight: kernelHeight,
             inputFeatureChannels: inputFeatureChannels,
             outputFeatureChannels: outputFeatureChannels)
        return buffer
    }
    
    static func makeBuffer(device: MTLDevice,
                           channelFormat: MPSImageFeatureChannelFormat,
                           outputFeatureChannels: Int,
                           biasTerms: UnsafePointer<Float>?) -> MTLBuffer {
        
        assert(channelFormat == .float16)
        
        let outputSlices = (outputFeatureChannels + 3) / 4
        let count = outputSlices * 4
        let buffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)
        
        
        if let biasTerms = biasTerms {
            copy(biasTerms: biasTerms, to: buffer, channelFormat: channelFormat,
                 outputFeatureChannels: outputFeatureChannels)
        }
        
        return buffer
    }
    
    static func copy(biasTerms: UnsafePointer<Float>,
                     to buffer: MTLBuffer,
                     channelFormat: MPSImageFeatureChannelFormat,
                     outputFeatureChannels: Int) {
        
        assert(channelFormat == .float16)
        
        let count = outputFeatureChannels
        assert(buffer.length / MemoryLayout<Float16>.stride >= count)
        
        let ptr = UnsafeMutablePointer(mutating: biasTerms)
        float32to16(input: ptr, output: buffer.contents(), count: count)
    }
    
    
    static func copy(weights: UnsafePointer<Float>,
                     to buffer: MTLBuffer,
                     channelFormat: MPSImageFeatureChannelFormat,
                     kernelWidth: Int,
                     kernelHeight: Int,
                     inputFeatureChannels: Int,
                     outputFeatureChannels: Int) {
        
        assert(channelFormat == .float16)
        
        let inputSlices = (inputFeatureChannels + 3) / 4
        let paddedInputChannels = inputSlices * 4
        let count = outputFeatureChannels * kernelHeight * kernelWidth * paddedInputChannels
        
        assert(buffer.length / MemoryLayout<Float16>.stride >= count)
        
        
        if paddedInputChannels == inputFeatureChannels {
            let ptr = UnsafeMutablePointer(mutating: weights)
            float32to16(input: ptr, output: buffer.contents(), count: count)
            
        } else {
            var srcPtr = UnsafeMutablePointer(mutating: weights)
            var dstPtr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
            
            for _ in 0..<outputFeatureChannels * kernelHeight * kernelWidth {
                float32to16(input: srcPtr, output: dstPtr, count: inputFeatureChannels)
                srcPtr += inputFeatureChannels
                dstPtr += paddedInputChannels
            }
        }
    }
    
    static func offsetForConvolution(padding: Int,
                                     sourceWidth: Int,
                                     sourceHeight: Int,
                                     destinationWidth: Int,
                                     destinationHeight: Int,
                                     kernelWidth: Int,
                                     kernelHeight: Int,
                                     strideInPixelsX: Int,
                                     strideInPixelsY: Int) -> MPSOffset {
        if padding > 0  {
            let padH = (destinationHeight - padding) * strideInPixelsY + kernelHeight - sourceHeight
            let padW = (destinationWidth  - padding) * strideInPixelsX + kernelWidth  - sourceWidth
            return MPSOffset(x: (kernelWidth - padW)/2, y: (kernelHeight - padH)/2, z: 0)
        } else {
            return MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
        }
    }
    
    var pipelines = [String: MTLComputePipelineState]()
    
    func getFunction(name: String, constants: [FuncConstantBase]? = nil) -> MTLComputePipelineState {
        let library = mdlLibrary
        if let pipelineState = pipelines[name] {
            return pipelineState
        }
        
        do {
            var function: MTLFunction
            if let constants = constants {
                let values = MTLFunctionConstantValues()
                for constant in constants {
                    var val = constant.getValue()
                    values.setConstantValue(&val, type: constant.type, at: constant.index)
                }
                function = try library.makeFunction(name: name, constantValues: values)
            } else {
                guard let loadedFunction = library.makeFunction(name: name) else {
                    fatalError("Function \(name) does not exist")
                }
                function = loadedFunction
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelines[name] = pipeline
            return pipeline
        } catch {
            fatalError("Unable to create pipeline state, check metal shaders")
        }
    }
    
    
}
