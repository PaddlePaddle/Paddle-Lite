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
class SplitLayer: Layer {
    
    override var type: String{
        return LayerModel.splitType
    }
    
    override func createImage(device: MTLDevice, commandBuffer: MTLCommandBuffer, infightIndex: Int) {
        (inputs[0].image as? MPSTemporaryImage)?.readCount = outputs.count
    }
    
    override func releaseImage() {
        inputs[0].image = nil
    }

    override func encode(commandBuffer: MTLCommandBuffer) {
        let input = inputs[0]
        for output in outputs {
            output.image = input.image
        }
    }
}




