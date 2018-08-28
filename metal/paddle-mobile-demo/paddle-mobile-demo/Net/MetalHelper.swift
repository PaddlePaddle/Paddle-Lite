//
//  MetalHelper.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/7/25.
//  Copyright © 2018年 orange. All rights reserved.
//

import Metal
import MetalKit
import Foundation
import paddle_mobile
import MetalPerformanceShaders

class MetalHelper {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let textureLoader: MTKTextureLoader
    static let shared: MetalHelper = MetalHelper.init()
    private init(){
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
        textureLoader = MTKTextureLoader.init(device: device)
    }
    
    static func scaleTexture(queue: MTLCommandQueue, input: MTLTexture, size:(width: Int, height: Int), complete: @escaping (MTLTexture) -> Void) {
        let tmpTextureDes = MTLTextureDescriptor.init()
        tmpTextureDes.width = size.width
        tmpTextureDes.height = size.height
        tmpTextureDes.depth = 1
        tmpTextureDes.usage = [.shaderRead, .shaderWrite]
        tmpTextureDes.pixelFormat = .rgba32Float
        tmpTextureDes.textureType = .type2D
        tmpTextureDes.storageMode = .shared
        tmpTextureDes.cpuCacheMode = .defaultCache
        let dest = MetalHelper.shared.device.makeTexture(descriptor: tmpTextureDes)
        
        let scale = MPSImageLanczosScale.init(device: MetalHelper.shared.device)
        
        let buffer = queue.makeCommandBuffer()
        scale.encode(commandBuffer: buffer!, sourceTexture: input, destinationTexture: dest!)
        buffer?.addCompletedHandler({ (buffer) in
            complete(dest!)
        })
        buffer?.commit()
    }
}

