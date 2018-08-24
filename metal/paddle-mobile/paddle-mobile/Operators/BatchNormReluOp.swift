//
//  BatchNormReluOp.swift
//  paddle-mobile
//
import Foundation

class BatchNormReluOp<P: PrecisionType>: Operator<BatchNormReluKernel<P>, BatchNormReluParam<P>>, Runable, Creator, InferShaperable, Fusion {
    
    typealias OpType = BatchNormReluOp<P>
    
    func runImpl(device: MTLDevice, buffer: MTLCommandBuffer) throws {
        do {
            try kernel.compute(commandBuffer: buffer, param: para)
        } catch let error {
            throw error
        }
    }
    
    static func fusionNode() -> Node {
        let beginNode = Node.init(inType: gConvType)
        _ = beginNode
            --> Node.init(inType: gBatchNormType)
            --> Node.init(inType: gReluType)
        return beginNode
    }
    
    static func change() -> [String : [(from: String, to: String)]] {
        return [:]
    }
    
    static func fusionType() -> String {
        return gBatchNormReluType
    }
    
    func inferShape() {
    }
    
}
