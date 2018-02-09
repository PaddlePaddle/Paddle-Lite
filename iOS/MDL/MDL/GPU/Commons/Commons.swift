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
/* The following copyright is cited from Forge as an acknowledgement for its inspiring framework.
 Copyright (c) 2016-2017 M.I. Hollemans
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to
 deal in the Software without restriction, including without limitation the
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 IN THE SOFTWARE.
 */




import Foundation
import Accelerate
import MetalPerformanceShaders


// 自定义 ?!  如果 ?! 前的返回值为一个可选值, 则进行隐式解包, 如果有值则返回这个值, 如果为nil 则fatalError 传入的信息
precedencegroup ExecutedOrFatalError{
    associativity: left
    higherThan: AssignmentPrecedence
}

infix operator ?!: ExecutedOrFatalError
func ?!<T>(option: T?, excuteOrError: @autoclosure () -> String) -> T{
    if let inOpt = option {
        return inOpt
    }else{
        fatalError(excuteOrError())
    }
}

@available(iOS 10.0, *)
class FileDataGetter <T>{
    let data: UnsafeMutablePointer<T>
    let dataSize: Int
    init(filePath: String) throws{
        let fileManager = FileManager.default
        guard let inData = fileManager.contents(atPath: filePath) else{
            throw NetError.loaderError(message: "fail to open file: " + filePath)
        }
        dataSize = inData.count
        data = UnsafeMutablePointer<T>.allocate(capacity: dataSize)
        let bufferPointer = UnsafeMutableBufferPointer(start: data, count: dataSize)
        let _ = inData.copyBytes(to: bufferPointer)
    }
    
    deinit {
        data.deinitialize()
        data.deallocate(capacity: dataSize)
    }
}


@available(iOS 10.0, *)
extension MPSImage{
    private func convert<T>(initial: T) -> [T] {
        
        //每个 image 的 slice 数量
        let numSlices = (featureChannels + 3)/4
        
        //将空白填充满后的 channels 数  1 2 4 4 8 8 8 8 ....
        let channelsPlusPadding = (featureChannels < 3) ? featureChannels : numSlices * 4
        
        //每一个slice的channel数
        let numComponents = (featureChannels < 3) ? featureChannels : 4
        
        //所占内存大小 (count  个 T 大小)
        let count = width * height * channelsPlusPadding * numberOfImages
        
        //初始化输出
        var output = [T](repeating: initial, count: count)
        
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: width, height: height, depth: 1))
        
        //numSlices*numberOfImages:   slice 总数
        for i in 0..<numSlices*numberOfImages {
            texture.getBytes(&(output[width * height * numComponents * i]),
                             bytesPerRow: width * numComponents * MemoryLayout<T>.stride,
                             bytesPerImage: 0,
                             from: region,
                             mipmapLevel: 0,
                             slice: i)
        }
        return output
    }
    
    private func fromFloat16() -> [Float] {
        var outputFloat16 = convert(initial: Float16(0))
        return float16to32(&outputFloat16, count: outputFloat16.count)
    }
    
    private func fromFloat32() -> [Float] {
        return convert(initial: Float(0))
    }
    
    @nonobjc func toFloatArray() -> [Float] {
        switch pixelFormat {
        case .r16Float, .rg16Float, .rgba16Float: return fromFloat16()
        case .r32Float, .rg32Float, .rgba32Float: return fromFloat32()
        default: fatalError("Pixel format \(pixelFormat) not supported")
        }
    }
    
    @nonobjc func toResultFloatArray() -> [Float]{
        var arr = toFloatArray()
        let loc = arr.count/50
        var result: [Float] = []
        for i in 0..<50 {
            result.append(arr[loc * i])
        }
        return result
    }
}

@available(iOS 10.0, *)
extension MTLComputeCommandEncoder{
    func dispatch(pipeline: MTLComputePipelineState, image: MPSImage) {
        dispatch(pipeline: pipeline,
                 width: image.width,
                 height: image.height,
                 featureChannels: image.featureChannels,
                 numberOfImages: image.numberOfImages)
    }
    
    
    func dispatch(pipeline: MTLComputePipelineState,
                  width: Int,
                  height: Int,
                  featureChannels: Int,
                  numberOfImages: Int = 1) {
        let slices = ((featureChannels + 3)/4) * numberOfImages
        
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let d = 1
        let threadGroupSize = MTLSizeMake(w, h, d)
        
        let threadGroups = MTLSizeMake(
            (width  + threadGroupSize.width  - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height,
            (slices + threadGroupSize.depth  - 1) / threadGroupSize.depth)
        setComputePipelineState(pipeline)
        dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
    
}

public typealias Float16 = UInt16

func float16to32(_ input: UnsafeMutablePointer<Float16>, count: Int) -> [Float] {
    var output = [Float](repeating: 0, count: count)
    float16to32(input: input, output: &output, count: count)
    return output
}

func float16to32(input: UnsafeMutablePointer<Float16>, output: UnsafeMutableRawPointer, count: Int) {
    var bufferFloat16 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 4)
    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
        print("Error converting float16 to float32")
    }
}

func float32to16(_ input: UnsafeMutablePointer<Float>, count: Int) -> [Float16] {
    var output = [Float16](repeating: 0, count: count)
    float32to16(input: input, output: &output, count: count)
    return output
}

func float32to16(input: UnsafeMutablePointer<Float>, output: UnsafeMutableRawPointer, count: Int) {
    var bufferFloat32 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 4)
    var bufferFloat16 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 2)
    if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
        print("Error converting float32 to float16")
    }
}

@available(iOS 10.0, *)
public protocol CustomKernel: Kernel{
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage)
}

@available(iOS 10.0, *)
public protocol Kernel{
    
}

@available(iOS 10.0, *)
extension MPSCNNConvolution: Kernel{
}


@available(iOS 10.0, *)
open class MPSKernel: CustomKernel {
    public let device: MTLDevice
    public let neuron: MPSCNNNeuron?
    
    public var offset = MPSOffset(x: 0, y: 0, z: 0)
    public var clipRect = MPSRectNoClip
    public var destinationFeatureChannelOffset = 0
    public var edgeMode = MPSImageEdgeMode.zero
    
    var params = KernelParams()
    
    @available(iOS 10.0, *)
    public init(device: MTLDevice, neuron: MPSCNNNeuron?, params: KernelParams) {
        self.device = device
        self.neuron = neuron
        self.params = params
        
    }
    
    public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        fatalError("Subclass must implement this function")
    }
}

@available(iOS 10.0, *)
public class DepthwiseConvolutionKernel: MPSKernel {
    let pipeline: MTLComputePipelineState
    let weightsBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
    
    public init(device: MTLDevice,
                kernelWidth: Int,
                kernelHeight: Int,
                featureChannels: Int,
                strideInPixelsX: Int = 1,
                strideInPixelsY: Int = 1,
                channelMultiplier: Int = 1,
                neuronFilter: MPSCNNNeuron?,
                kernelWeights: UnsafePointer<Float>,
                biasTerms: UnsafePointer<Float>?) {
        
        precondition(kernelWidth == 3 && kernelHeight == 3, "Only 3x3 kernels are currently supported")
        precondition(channelMultiplier == 1, "Channel multipliers are not supported yet")
        
        let inputSlices = (featureChannels + 3) / 4
        let paddedInputChannels = inputSlices * 4
        let count = kernelHeight * kernelWidth * paddedInputChannels
        weightsBuffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)
        
        MetalManager.copy(weights: kernelWeights, to: weightsBuffer, channelFormat: .float16,
                          kernelWidth: kernelWidth, kernelHeight: kernelHeight,
                          inputFeatureChannels: featureChannels, outputFeatureChannels: 1)
        
        biasBuffer = MetalManager.makeBuffer(device: device,
                                             channelFormat: .float16,
                                             outputFeatureChannels: featureChannels,
                                             biasTerms: biasTerms)
        
        var params = KernelParams()
        let constants = MTLFunctionConstantValues()
        configureNeuronType(filter: neuronFilter, constants: constants, params: &params)
        
        var stride = [ UInt16(strideInPixelsX), UInt16(strideInPixelsY) ]
        constants.setConstantValue(&stride, type: .ushort2, withName: "stride")
        
        let functionName: String
        if featureChannels <= 4 {
            functionName = "depthwiseConv3x3"
        } else {
            functionName = "depthwiseConv3x3_array"
        }
        pipeline = MetalManager.makeFunction(device: device, name: functionName,
                                             constantValues: constants)
        
        super.init(device: device, neuron: neuronFilter, params: params)
    }
    
    public override func encode(commandBuffer: MTLCommandBuffer,
                                sourceImage: MPSImage, destinationImage: MPSImage) {
        params.inputOffsetX = Int16(offset.x);
        params.inputOffsetY = Int16(offset.y);
        params.inputOffsetZ = Int16(offset.z);
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.setBytes(&params, length: MemoryLayout<KernelParams>.size, at: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, at: 1)
        encoder.setBuffer(biasBuffer, offset: 0, at: 2)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}

@available(iOS 10.0, *)
open class MetalKernel {
    let device: MTLDevice
    let pipeline: MTLComputePipelineState
    let name: String
    
    public init(device: MTLDevice, functionName: String, bundle: Bundle,useMmsLibrary: Bool = false) {
        self.device = device
        self.name = functionName
        pipeline = MetalManager.makeFunction(device: device, name: functionName, bundle: bundle)
    }
    
    public func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.pushDebugGroup(name)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.popDebugGroup()
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}

@available(iOS 10.0, *)
extension MetalKernel: CustomKernel{
}

@available(iOS 10.0, *)
func configureNeuronType(filter: MPSCNNNeuron?,
                         constants: MTLFunctionConstantValues,
                         params: inout KernelParams) {
    var neuronType: UInt16 = 0
    if let filter = filter as? MPSCNNNeuronReLU {
        neuronType = 1
        params.neuronA = filter.a
    } else if let filter = filter as? MPSCNNNeuronLinear {
        neuronType = 2
        params.neuronA = filter.a
        params.neuronB = filter.b
    } else if filter is MPSCNNNeuronSigmoid {
        neuronType = 3
    } else if let filter = filter as? MPSCNNNeuronTanH {
        neuronType = 4
        params.neuronA = filter.a
        params.neuronB = filter.b
    } else if filter is MPSCNNNeuronAbsolute {
        neuronType = 5
    }
    constants.setConstantValue(&neuronType, type: .ushort, withName: "neuronType")
}


public struct KernelParams {
    var inputWidth: UInt16 = 0
    var inputHeight: UInt16 = 0
    var inputFeatureChannels: UInt16 = 0
    var inputSlices: UInt16 = 0
    
    var inputOffsetX: Int16 = 0
    var inputOffsetY: Int16 = 0
    var inputOffsetZ: Int16 = 0
    var outputWidth: UInt16 = 0
    var outputHeight: UInt16 = 0
    var outputFeatureChannels: UInt16 = 0
    var outputSlices: UInt16 = 0
    var destinationSliceOffset: UInt16 = 0
    var outputOffsetX: Int16 = 0
    var outputOffsetY: Int16 = 0
    var outputOffsetZ: Int16 = 0
    var edgeMode: UInt16 = 0
    var neuronA: Float = 0
    var neuronB: Float = 0
}


public enum NetParameterType {
    case weights
    case biases
}

/// 存储着网络层运算所需的 权重 或者 偏置
///store the weight or biase of the net layer
public protocol NetParameterData {
    var pointer: UnsafeMutablePointer<Float> { get }
}

public class ParameterData: NetParameterData{
    public var pointer: UnsafeMutablePointer<Float>
    let size: Int
    init(size: Int) {
        self.size = size
        pointer = UnsafeMutablePointer<Float>.allocate(capacity: size)
        pointer.initialize(to: 0.0)
    }
    deinit {
        pointer.deinitialize()
        pointer.deallocate(capacity: size)
    }
}


public class NetParameterLoaderBundle: NetParameterData {
    private var fileSize: Int
    private var fd: CInt!
    private var hdr: UnsafeMutableRawPointer!
    private(set) public var pointer: UnsafeMutablePointer<Float>
    
    public init?(name: String, count: Int, ext: String, bundle: Bundle = Bundle.main) {
        fileSize = count * MemoryLayout<Float>.stride
        guard let path = bundle.path(forResource: name, ofType: ext) else {
            print("Parameter Get Error: resource \"\(name)\" not found")
            return nil
        }
        
        fd = open(path, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        if fd == -1 {
            print("Parameter Get Error: failed to open \"\(path)\", error = \(errno)")
            return nil
        }
        
        hdr = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
        if hdr == nil {
            print("Parameter Get Error: mmap failed, errno = \(errno)")
            return nil
        }
        
        pointer = hdr.bindMemory(to: Float.self, capacity: count)
        if pointer == UnsafeMutablePointer<Float>(bitPattern: -1) {
            print("Parameter Get Error: mmap failed, errno = \(errno)")
            return nil
        }
    }
    
    deinit {
        if let hdr = hdr {
            let result = munmap(hdr, Int(fileSize))
            assert(result == 0, "Parameter Get Error: munmap failed, errno = \(errno)")
        }
        if let fd = fd {
            close(fd)
        }
    }
}





