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

/// 全链接层 Full connect
@available(iOS 10.0, *)
class FcLayer: MPSCNNLayer {
    let outputNum: Int
    var activation: MPSCNNNeuron?
    var userBiase: Bool{
        return weights.count >= 2
    }
    
    override init(device: MTLDevice,
                config: LayerModel) throws{
        guard let inOutputNum = config.param?.output_num else {
            throw NetError.modelDataError(message: "full connect need neurons")
        }
        self.outputNum = inOutputNum
        try super.init(device:device, config: config)
    }
    
    override var type: String {
        return LayerModel.fcType
    }
        
    override func initializeCompute(device: MTLDevice) {
        let input = inputs[0]
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: input.width,
                                               kernelHeight: input.height,
                                               inputFeatureChannels: input.channels,
                                               outputFeatureChannels: outputNum,
                                               neuronFilter: activation)
        
        var biasTerms: UnsafeMutablePointer<Float>?
        if self.userBiase {
            guard let biases = weights[1].data else {
                fatalError("偏差为空")
            }
            biasTerms = biases.pointer
        }
        
        let weight = weights[0]
        mpscnn = MPSCNNFullyConnected(device: device,
                                      convolutionDescriptor: desc,
                                      kernelWeights: weight.data!.pointer,
                                      biasTerms: biasTerms,
                                      flags: .none)
    }
}
