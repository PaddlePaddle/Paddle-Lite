//
//  TestConvAddBatchNormRelu.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/7/25.
//  Copyright © 2018年 orange. All rights reserved.
//

import Metal
import Foundation
import MetalPerformanceShaders

@available(iOS 10.0, *)
public class PaddleMobileUnitTest {
    let device: MTLDevice
    let queue: MTLCommandQueue
//    var imageDes:MPSImageDescriptor
//    var image: MPSImage
    
    
    
    public init(inDevice: MTLDevice, inQueue: MTLCommandQueue) {
        device = inDevice
        queue = inQueue
    }
    
    
    public func testConvAddBnRelu() {
        let buffer = queue.makeCommandBuffer() ?! " buffer is nil "
        
        let input: [Float32] = [
         1.0, 2.0, 3.0, 4.0,
         1.0, 2.0, 3.0, 4.0,
         1.0, 2.0, 3.0, 4.0,
         
         1.0, 2.0, 3.0, 4.0,
         1.0, 2.0, 3.0, 4.0,
         1.0, 2.0, 3.0, 4.0,
         
         1.0, 2.0, 3.0, 4.0,
         1.0, 2.0, 3.0, 4.0,
         1.0, 2.0, 3.0, 4.0,
        ]
        
        let filter: [Float32] = [
        //1.0
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        //2.0
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        //3.0
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        //4.0
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        ]
        
        let biase: [Float32] = [1.0, 1.0, 1.0, 100.0]
        let newScalue: [Float32] = [1.0, 1.0, 1.0, 1.0]
        let newBiase: [Float32] = [1.0, 1.0, 1.0, 1.0]
        
        let inputeTexture = device.makeFloatTexture(value: input, textureWidth: 3, textureHeight: 3, arrayLength: 1)
        
        //filter
        let filterBuffer = device.makeBuffer(value: filter)
        
        // biase
        let biaseBuffer = device.makeBuffer(value: biase)
        
        // new scale
        let newScalueBuffer = device.makeBuffer(value: newScalue)
        
        // new biase
        let newBiaseBuffer = device.makeBuffer(value: newBiase)
        
        //output
        let outputTexture = device.makeFloatTexture(value: [Float32](), textureWidth: 2, textureHeight: 2, arrayLength: 1)
        
        let filterSize: (width: Int, height: Int, channel: Int) = (3, 3, 4)
        let paddings: (Int, Int) = (1, 1)
        let stride: (Int, Int) = (2, 2)
        
        let offsetX = filterSize.width/2 - paddings.0
        let offsetY = filterSize.height/2 - paddings.1
        
        let metalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: 0, strideX: UInt16(stride.0), strideY: UInt16(stride.1), paddedZ: UInt16(paddings.0))
        
        let param = ConvAddBatchNormReluTestParam.init(inInputTexture: inputeTexture, inOutputTexture: outputTexture, inMetalParam: metalParam, inFilterBuffer: filterBuffer, inBiaseBuffer: biaseBuffer, inNewScaleBuffer: newScalueBuffer, inNewBiaseBuffer: newBiaseBuffer, inFilterSize: filterSize)
        
        
        let convAddBnReluKernel = ConvAddBatchNormReluKernel<Float32>.init(device: device, testParam: param)
        
        convAddBnReluKernel.test(commandBuffer: buffer, param: param)
        
        buffer.addCompletedHandler { (buffer) in
            let _: Float32? = inputeTexture.logDesc(header: "input texture", stridable: false)
            let _: Float32? = outputTexture.logDesc(header: "output texture", stridable: false)
        }
        
        buffer.commit()
        
        
//        let inputTexture = device.makeFloatTexture(value: <#T##[P]#>, textureWidth: <#T##Int#>, textureHeight: <#T##Int#>, arrayLength: <#T##Int#>)
        
        
//        let param = ConvAddBatchNormReluTestParam.init(inInputTexture: <#T##MTLTexture#>, inOutputTexture: <#T##MTLTexture#>, inMetalParam: <#T##MetalConvParam#>, inFilterBuffer: <#T##MTLBuffer#>, inBiaseBuffer: <#T##MTLBuffer#>, inNewScaleBuffer: <#T##MTLBuffer#>, inNewBiaseBuffer: <#T##MTLBuffer#>, inFilterSize: <#T##(width: Int, height: Int, channel: Int)#>)
        
//        ConvAddBatchNormReluKernel.init(device: <#T##MTLDevice#>, testParam: <#T##ConvAddBatchNormReluTestParam#>)
        
        
    }
}



