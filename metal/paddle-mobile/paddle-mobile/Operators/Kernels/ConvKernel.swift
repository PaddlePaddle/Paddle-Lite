//
//  ConvKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


class ConvKernel<P: PrecisionType>: Kernel, Computable {
    func compute(commandBuffer: MTLCommandBuffer, param: ConvParam<P>) throws {
       
    }
    required init(device: MTLDevice) {
        super.init(device: device, inFunctionName: "conv")
    }
    
}
