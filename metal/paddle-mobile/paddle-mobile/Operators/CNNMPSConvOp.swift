//
//  CNNConvAddBatchNormReluOp.swift
//  paddle-mobile

import Foundation

class CNNMPSConvTestParam: TestParam {
    var outputTexture: MTLTexture?
    var metalParam: MetalConvParam
    let filterPointer: UnsafeMutableRawPointer
    let biasePointer: UnsafeMutablePointer<Float>
    let filterSize: (width: Int, height: Int, channel: Int)
    init(inMetalParam: MetalConvParam, inFilter: [Float], inBiase: [Float], inFilterSize: (width: Int, height: Int, channel: Int)) {
        metalParam = inMetalParam
        filterPointer = UnsafeMutableRawPointer.init(mutating: inFilter)
        biasePointer = UnsafeMutablePointer.init(mutating: inBiase)
        filterSize = inFilterSize
    }
}

@available(iOS 10.0, *)
class CNNMPSConvOp<P: PrecisionType>: Operator<CNNConvKernel<P>, CNNConvParam<P>>, Runable, Creator, InferShaperable, Fusion {
    
    typealias OpType = CNNMPSConvOp<P>

    required init(device: MTLDevice, opDesc: OpDesc, inScope: Scope) throws {
        fatalError()
    }
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        do {
            try kernel.compute(commandBuffer: buffer, param: para)
        } catch let error {
            throw error
        }
    }
    
    func delogOutput() {
    }
    
    static func fusionNode() -> Node {
        let beginNode = Node.init(inType: gConvType)
        _ = beginNode-->Node.init(inType: gElementwiseAdd);
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    static func fusionType() -> String {
        return gMPSCNNConvType
    }
    func inferShape() {
        let inDims = para.input.dim
        let filterDim = para.filter.dim
        let strides = para.stride
        let paddings = para.paddings
        let dilations = para.dilations
        
        var outDim = [inDims[0]]
        for i in 0..<strides.count {
            let dilation: Int = Int(dilations[i])
            let filterSize: Int = filterDim[i + 1]
            let inputSize: Int = inDims[i + 1]
            let padding: Int = Int(paddings[i])
            let stride: Int = Int(strides[i])
            let dKernel = dilation * (filterSize - 1) + 1
            let outputSize = (inputSize + 2 * padding - dKernel) / stride + 1
            outDim.append(outputSize)
        }
        outDim.append(filterDim[0])
        para.output.dim = Dim.init(inDim: outDim)
    }
}
