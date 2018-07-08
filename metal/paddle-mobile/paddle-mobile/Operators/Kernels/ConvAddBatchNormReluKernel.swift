//
//  ConvKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

class ConvAddBatchNormReluKernel<P: PrecisionType>: Kernel, Computable {
    required init(device: MTLDevice, param: ConvParam<P>) {
        super.init(device: device, inFunctionName: "conv3x3")
        
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConvParam<P>) throws {
    }
}
