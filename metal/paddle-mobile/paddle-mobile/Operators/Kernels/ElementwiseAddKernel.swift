//
//  ElementwiseAddKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


class ElementwiseAddKernel<P: PrecisionType>: Kernel, Computable {
    required init(device: MTLDevice) {
        super.init(device: device, inFunctionName: "conv")
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ElementwiseAddParam<P>) throws {
        
    }
}
