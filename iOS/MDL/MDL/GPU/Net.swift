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

struct NetResult{
    var resultArr: [Float] = []
    var elapsedTime: TimeInterval = 0
}

@available(iOS 10.0, *)
class Net {
    var inflightIndex = 0
    var inflightBuffers = 3
    var model: Model
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    var layers: [Layer] = []
    let inflightSemaphore: DispatchSemaphore
    var descriptorList: [Matrix : MPSImageDescriptor] = [:]
    var outputLayer: Layer?
    var images: [(String, MPSImage)] = []
    
    init(device: MTLDevice = MTLCreateSystemDefaultDevice()!, model: Model, inflightBuffers: Int = 3, preProcessKernel: CustomKernel?, commandQueue: MTLCommandQueue) throws{
        guard model.layer.count > 0, model.matrix.count > 0 else {
            throw NetError.modelDataError(message: "the count of layer or matrix in model mustn't be 0")
        }
        self.inflightBuffers = inflightBuffers
        self.inflightSemaphore = DispatchSemaphore(value: inflightBuffers)
        self.device = device
        self.model = model
        self.commandQueue = commandQueue
        self.descriptorList = model.matrixDesDic
        if preProcessKernel != nil {
            self.ensurePreprocessLayer(config: model)
        }
        
        self.ensureFirstLayer(config: model)
        
        var layer: Layer
        for layerConfig in model.layer {
            switch layerConfig.type {
            case LayerModel.resizeType:
                try layer = ResizeLayer(device: device, config: layerConfig)
            case LayerModel.convolutionType:
                try layer = ConvolutionLayer(device: device, config: layerConfig)
            case LayerModel.depthWiseConvolutionType:
                try layer = DepthwiseConvolution(device: device, config: layerConfig)
            case LayerModel.reluType:
                try layer = ReluLayer(device: device, config: layerConfig)
            case LayerModel.averagePoolingType:
                try layer = AveragePooling(device: device, config: layerConfig)
            case LayerModel.softmaxType:
                try layer = SoftmaxLayer(device: device, config: layerConfig)
            case LayerModel.poolType:
                try layer = PoolingLayer(device: device, config: layerConfig)
            case LayerModel.fcType:
                try layer = FcLayer(device: device, config: layerConfig)
            case LayerModel.pointWiseType:
                try layer = PointwiseConvolutionLayer(device: device, config: layerConfig)
            case LayerModel.globalAveragePoolingType:
                try layer = GlobalAveragePooling(device: device, config: layerConfig)
            case LayerModel.splitType:
                try layer = SplitLayer(device: device, config: layerConfig)
            case LayerModel.concatType:
                try layer = ConcatLayer(device: device, config: layerConfig)
            case LayerModel.activationType:
                try layer = ActivationLayer(device: device, config: layerConfig)
            case LayerModel.maxPool:
                try layer = MaxPooling(device: device, config: layerConfig)
            case LayerModel.preProcessType:
                guard let inPreProcessKernel = preProcessKernel else{
                    throw NetError.netError(message: "need custom kernel")
                }
                try layer = PreProcessLayer(inPreProcessKernel, device: device, config: layerConfig)
            default:
                throw NetError.modelDataError(message: "unknown layer type: " + layerConfig.type)
            }
            
            layer.initializeCompute(device: device)
            layers.append(layer)
        }
        layers.first?.firstLayer = true
        layers.last?.lastLayer = true
        layers.last?.createImages(device: device, infightBuffers: inflightBuffers)
        outputLayer = layers.last
    }
    
    private func ensureFirstLayer(config: Model){
        guard let firstLayer = config.layer.first, firstLayer.type != LayerModel.resizeType else {
            return
        }
        let layerModel = LayerModel()
        layerModel.name = "Resize"
        layerModel.output = [firstLayer.input[0]]
        layerModel.outputMatrices = [firstLayer.inputMatrices[0]]
        layerModel.type = LayerModel.resizeType
        config.layer.insert(layerModel, at: 0)
        _ = layerModel.outputMatrices[0].createImageDes()
    }
    private func ensurePreprocessLayer(config: Model){
        guard let firstLayer = config.layer.first, firstLayer.type != LayerModel.preProcessType else {
            return
        }
        
        if config.layer.count >= 2 && config.layer[1].type == LayerModel.preProcessType{
            return
        }
        let matrix: Matrix = Matrix.init(device: device, name: "inPre", config: firstLayer.inputMatrices[0].config)
        _ = matrix.createImageDes()
        let layerModel = LayerModel()
        layerModel.name = "PreProcess"
        layerModel.output = [firstLayer.input[0]]
        layerModel.outputMatrices = [firstLayer.inputMatrices[0]]
        layerModel.inputMatrices = [matrix]
        layerModel.input = ["PreProcess"]
        layerModel.type = LayerModel.preProcessType
        config.layer.insert(layerModel, at: 0)
        _ = layerModel.outputMatrices[0].createImageDes()
    }
    
    func predict(inputTexture: MTLTexture, queue: DispatchQueue, threadNum: Int = 3, completion: @escaping (NetResult) -> Void) {
        let date = Date()
        inflightSemaphore.wait()
        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()
            let values = Array(descriptorList.values)
            MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: values)
            for layer in layers {
                layer.createImage(device: device, commandBuffer: commandBuffer, infightIndex: inflightIndex)
                if layer.firstLayer {
                    layer.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture)
                }else{
                    layer.encode(commandBuffer: commandBuffer)
                }
                layer.releaseImage()
            }
            
            commandBuffer.addCompletedHandler({[inflightIndex] commandBuffer in
                let resultArr = self.outputLayer?.outputs[0].images[inflightIndex].toFloatArray() ?? []
                var result = NetResult()
                result.resultArr = resultArr
                result.elapsedTime = Date().timeIntervalSince(date)
                queue.async {completion(result)}
                
                self.inflightSemaphore.signal()
            })
            inflightIndex = (inflightIndex + 1) % inflightBuffers
            commandBuffer.commit()
        }
    }
}




