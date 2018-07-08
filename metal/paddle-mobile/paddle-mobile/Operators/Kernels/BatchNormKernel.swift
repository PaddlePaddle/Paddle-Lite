//
//  BatchNormKernel.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/5.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

class BatchNormKernel<P: PrecisionType>: Kernel, Computable {
    required init(device: MTLDevice, param: BatchNormParam<P>) {
        super.init(device: device, inFunctionName: "batchnorm")
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: BatchNormParam<P>) throws {
        
    }
}
