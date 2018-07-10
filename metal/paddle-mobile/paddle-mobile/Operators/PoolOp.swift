//
//  PoolOp.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/9.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

class PoolParam<P: PrecisionType>: OpParam {
    typealias ParamPrecisionType = P
    required init(opDesc: OpDesc, inScope: Scope) throws {
        do {
            input = try PoolParam.inputX(inputs: opDesc.inputs, from: inScope)
            output = try PoolParam.outputOut(outputs: opDesc.outputs, from: inScope)
        } catch let error {
            throw error
        }
    }
    let input: Texture<P>
    var output: Texture<P>
}

class PoolOp<P: PrecisionType>: Operator<PoolKernel<P>, PoolParam<P>>, Runable, Creator, InferShaperable{
    
    func inferShape() {
        para.output.dim = para.input.dim
    }
    
    typealias OpType = PoolOp<P>
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        do {
            try kernel.compute(commandBuffer: buffer, param: para)
        } catch let error {
            throw error
        }
    }
}
