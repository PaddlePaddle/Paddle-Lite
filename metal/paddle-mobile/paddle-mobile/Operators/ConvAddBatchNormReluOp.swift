//
//  ConvAddBatchNormReluOp.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/8.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


class ConvAddBatchNormReluOp<P: PrecisionType>: Operator<ConvAddBatchNormReluKernel<P>, ConvParam<P>>, Runable, Creator, InferShaperable, Fusion{
    static func fusionNode() -> Node {
        let beginNode = Node.init(inType: gConvType)
        _ = beginNode
            --> Node.init(inType: gElementwiseAdd)
            --> Node.init(inType: gBatchNormType)
            --> Node.init(inType: gReluType)
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    typealias OpType = ConvAddBatchNormReluOp<P>
    
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

    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        do {
            try kernel.compute(commandBuffer: buffer, param: para)
        } catch let error {
            throw error
        }
    }
    
}
