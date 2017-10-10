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

@available(iOS 10.0, *)
class Layer {
    var useBias: Bool{
        return self.weights.count > 1
    }
    
    var firstLayer = false
    var lastLayer = false
    var pid: Int?
    var config: LayerModel
    
    internal(set) var name: String
    internal(set) var inputs: [Matrix] = []
    internal(set) var outputs: [Matrix] = []
    internal(set) var weights: [Matrix] = []

    init(device: MTLDevice, config: LayerModel) throws {
        self.config = config
        self.pid = config.pid
        self.name = config.name

        inputs = config.inputMatrices
        outputs = config.outputMatrices
        
        for weight in config.weight {
            if let matrix = Loader.share.matrices[weight]{
                matrix.matrixType = .kernel
                matrix.name = weight
                weights.append(matrix)
            }
        }
    }
    
    func createImages(device: MTLDevice, infightBuffers: Int) {
        if lastLayer{
            for matrix in outputs {
                matrix.creatImages(device: device, infightBuffers: infightBuffers)
            }
        }
    }
    
    func createImage(device: MTLDevice, commandBuffer: MTLCommandBuffer, infightIndex: Int) {
        if lastLayer{
            for matrix in outputs {
                matrix.image = matrix.images[infightIndex]
                (matrix.image as? MPSTemporaryImage)?.readCount = 1
            }
        }else{
            for matrix in outputs {
                matrix.createTemporaryImage(commandBuffer: commandBuffer)
                (matrix.image as? MPSTemporaryImage)?.readCount = 1
            }
        }
        
    }
    
    func releaseImage() {
        for matrix in inputs{
            if let temporaryImage = matrix.image as? MPSTemporaryImage {
                if temporaryImage.readCount > 0{
                }else{
                    matrix.image = nil
                }
            }
        }
    }
    
    
    func createImageDes(created: (Matrix, MPSImageDescriptor) -> Void) {
        for matrix in inputs {
            if let imageDes = matrix.createImageDes(){
                created(matrix, imageDes)
            }
        }
    }
    
    func encode(commandBuffer: MTLCommandBuffer,
                sourceTexture: MTLTexture){
        fatalError("Subclass must implement this function")
    }
    func encode(commandBuffer: MTLCommandBuffer){
        fatalError("Subclass must implement this function")
    }
    func initializeCompute(device: MTLDevice){
    }
    
    open var type: String {
        fatalError("Subclass must implement this function")
    }
}

@available(iOS 10.0, *)
class MPSCNNLayer: Layer {
    var destinationChannelOffset = 0
    var mpscnn: MPSCNNKernel!
    override init(device: MTLDevice, config: LayerModel) throws {
        guard config.input.count > 0 && config.output.count > 0 else {
            throw NetError.modelDataError(message: "input or output has no element" + config.name)
        }
        destinationChannelOffset = config.destinationChannelOffset
        try super.init(device: device, config: config)
    }
    
    override func encode(commandBuffer: MTLCommandBuffer) {
        guard inputs.count > 0 && outputs.count > 0 else {
            fatalError("inputs or outpus has no element")
        }
        let input = inputs[0]
        let output = outputs[0]
        guard let inImage = input.image, let outImage = output.image else {
            fatalError("input image or output image is nil")
        }
        mpscnn.destinationFeatureChannelOffset = destinationChannelOffset
        mpscnn.encode(commandBuffer: commandBuffer,
                      sourceImage: inImage,
                      destinationImage: outImage)
    }
}
