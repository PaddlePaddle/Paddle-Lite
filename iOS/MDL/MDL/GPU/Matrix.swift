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

enum MatrixType {
    case layer
    case kernel
    case bias
}

@available(iOS 10.0, *)
class Matrix{
    var matrixType: MatrixType?
    var number = 1
    var width = 1
    var height = 1
    var channels = 1
    var name = ""
    var imageDes: MPSImageDescriptor?
    var image: MPSImage?
    var images: [MPSImage] = []
    let config: [Int]
    var concatMatrix: Matrix?
    init(device: MTLDevice, name: String, config: [Int]) {
        self.name = name
        self.config = config
        if config.count == 4 {
            width = config[2]
            height = config[3]
            channels = config[1]
            number = config[0]
        } else if config.count == 2{
            number = config[0]
            channels = config[1]
            width = 1
            height = 1
        }
    }
    
    func createImageDes() -> MPSImageDescriptor?{
        guard imageDes == nil else {
            return imageDes
        }
        imageDes = MPSImageDescriptor(channelFormat: .float16, width: width, height: height, featureChannels: channels)
        imageDes?.storageMode = .private
        return imageDes
    }
    
    func creatImages(device: MTLDevice, infightBuffers: Int){
        imageDes = MPSImageDescriptor(channelFormat: .float16, width: width, height: height, featureChannels: channels)
        imageDes?.storageMode = .shared
        for _ in 0..<infightBuffers {
            images.append(MPSImage(device: device, imageDescriptor: imageDes!))
        }
    }
    
    func createTemporaryImage(commandBuffer: MTLCommandBuffer){
        let des = imageDes ?! "image des is nil"
        
        if let concatMatrix = self.concatMatrix {
            concatMatrix.createTemporaryImage(commandBuffer: commandBuffer)
            self.image = concatMatrix.image
            return
        }
        guard image == nil else {
            return
        }
        image = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: des)
    }
    
    var data: NetParameterData?
    
    func count() -> Int {
        return count(begin: 0, end: config.count)
    }
    
    func count(begin: Int, end: Int) -> Int {
        return config[begin ..< end].reduce(1) { $0 * $1 }
    }
}

@available(iOS 10.0, *)
extension Matrix: Hashable{
    static func ==(lhs: Matrix, rhs: Matrix) -> Bool {
        return lhs.width == rhs.width && lhs.height == lhs.height && lhs.channels == rhs.channels
    }

    var hashValue: Int{
        return width + height*1000 + channels*1000*1000
    }
}



