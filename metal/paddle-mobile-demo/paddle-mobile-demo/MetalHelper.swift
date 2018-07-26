//
//  MetalHelper.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/7/25.
//  Copyright © 2018年 orange. All rights reserved.
//

import Metal
import paddle_mobile
import Foundation


class MetalHelper {
    let device: MTLDevice
    let queue: MTLCommandQueue
    static let shared: MetalHelper = MetalHelper.init()
    private init(){
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
    }
}

