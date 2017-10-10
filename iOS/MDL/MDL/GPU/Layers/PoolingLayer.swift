/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 ==============================================================================*/


import Foundation
import MetalPerformanceShaders


/// 池化层
@available(iOS 10.0, *)
class PoolingLayer: MPSCNNLayer {
    let kernel: (Int, Int)
    let stride: (Int, Int)
    let pad: Int
    let edgeMode: MPSImageEdgeMode
    var pool: MPSCNNPooling!
    var poolType: String
    
    override init(device: MTLDevice, config: LayerModel) throws {
        self.kernel = (config.param?.kernel_size ?? 0, config.param?.kernel_size ?? 0)
        self.stride = (config.param?.stride ?? 0, config.param?.stride ?? 0 )
        self.pad = config.param?.pad ?? 0
        self.edgeMode = .clamp
        self.poolType = config.param?.type ?? ""
        
        try super.init(device: device, config: config)
    }
    
    func offsetForPooling(pad: Int,
                          sourceWidth: Int,
                          sourceHeight: Int,
                          kernelWidth: Int,
                          kernelHeight: Int,
                          strideInPixelsX: Int,
                          strideInPixelsY: Int) -> MPSOffset {
        if pad > 0 {
            var offset = MPSOffset(x: 0, y: 0, z: 0)
            if kernelWidth % 2 == 0 {
                offset.x += (((sourceWidth - 1) % strideInPixelsX) / 2) + 1
            } else {
                offset.x += (((sourceWidth - 1) % strideInPixelsX) + 1) / 2
            }
            if kernelHeight % 2 == 0 {
                offset.y += (((sourceHeight - 1) % strideInPixelsY) / 2) + 1
            } else {
                offset.y += (((sourceHeight - 1) % strideInPixelsY) + 1) / 2
            }
            return offset
        } else {
            return MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
        }
    }

    override func initializeCompute(device: MTLDevice) {
        if config.param?.global_pooling ?? false{
            initializeGlobalCompute(device: device)
            return
        }
        if poolType == "max"{
            initializeMaxPoolingCompute(device: device)
        }else if poolType == "ave" {
            initializeAvcPoolingCompute(device: device)
        }else{
            fatalError("need pool layer type")
        }
    }
    
    func initializeGlobalCompute(device: MTLDevice){
        let input = inputs[0]
        let _ = outputs[0]
        self.pool = MPSCNNPoolingAverage(device: device,
                                        kernelWidth: input.width,
                                        kernelHeight: input.height,
                                        strideInPixelsX: input.width,
                                        strideInPixelsY: input.height)
        pool.offset = MPSOffset(x: input.width/2, y: input.height/2, z: 0)
        pool.edgeMode = .clamp
        mpscnn = pool
    }
    
    fileprivate func initializeMaxPoolingCompute(device: MTLDevice){
        pool = MPSCNNPoolingMax(device: device,
                                kernelWidth: kernel.0,
                                kernelHeight: kernel.1,
                                strideInPixelsX: stride.0,
                                strideInPixelsY: stride.1)
        pool.edgeMode = edgeMode
        mpscnn = pool
    }
    
    fileprivate func initializeAvcPoolingCompute(device: MTLDevice){
        pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: kernel.0,
                                    kernelHeight: kernel.1,
                                    strideInPixelsX: stride.0,
                                    strideInPixelsY: stride.1)
        pool.edgeMode = edgeMode
        mpscnn = pool
    }
    override var type: String {
        return "PoolingLayer"
    }
    override  func encode(commandBuffer: MTLCommandBuffer){
        let input = inputs[0]
        if !(config.param?.global_pooling ?? false){
            pool.offset = offsetForPooling(pad: self.pad,
                                           sourceWidth: input.width,
                                           sourceHeight: input.height,
                                           kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           strideInPixelsX: stride.0,
                                           strideInPixelsY: stride.1)
        }
        super.encode(commandBuffer: commandBuffer)
    }
}

@available(iOS 10.0, *)
class MaxPooling: PoolingLayer {
    override var type: String {
        return "MaxPool"
    }
    
    override func initializeCompute(device: MTLDevice) {
        initializeMaxPoolingCompute(device: device)
    }
}

@available(iOS 10.0, *)
class AveragePooling: PoolingLayer {
    override var type: String {
        return "AveragePooling"
    }
    override func initializeCompute(device: MTLDevice){
        initializeAvcPoolingCompute(device: device)
    }
}

@available(iOS 10.0, *)
class GlobalAveragePooling: MPSCNNLayer {
    override var type: String {
        return LayerModel.globalAveragePoolingType
    }
    
    override func initializeCompute(device: MTLDevice){
        let input = inputs[0]
        let _ = outputs[0]
        let pool = MPSCNNPoolingAverage(device: device,
                                        kernelWidth: input.width,
                                        kernelHeight: input.height,
                                        strideInPixelsX: input.width,
                                        strideInPixelsY: input.height)
        pool.offset = MPSOffset(x: input.width/2, y: input.height/2, z: 0)
        pool.edgeMode = .clamp
        self.mpscnn = pool
    }
}

