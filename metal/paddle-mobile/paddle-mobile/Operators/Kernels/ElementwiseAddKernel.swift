//
//  ElementwiseAddKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation


class ElementwiseAddKernel<P: PrecisionType>: Kernel, Computable {
    required init(device: MTLDevice, param: ElementwiseAddParam<P>) {
        super.init(device: device, inFunctionName: "elementwise_add")
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ElementwiseAddParam<P>) throws {
        
    }
}
