/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

import UIKit
import MetalKit
import paddle_mobile
import MetalPerformanceShaders

let openTest: Bool = false

class PreProccess: CusomKernel {
    init(device: MTLDevice) {
        let s = CusomKernel.Shape.init(inWidth: 224, inHeight: 224, inChannel: 3)
        super.init(device: device, inFunctionName: "preprocess", outputDim: s, usePaddleMobileLib: false)
    }
}

class ViewController: UIViewController {
    var textureLoader: MTKTextureLoader!
    var program: Program!
    var executor: Executor<Float32>!
    var preprocessKernel: PreProccess!
    
//    let queue: MTLCommandQueue
    func scaleTexture(queue: MTLCommandQueue, input: MTLTexture, complete: @escaping (MTLTexture) -> Void) {        
        let tmpTextureDes = MTLTextureDescriptor.init()
        tmpTextureDes.width = 224
        tmpTextureDes.height = 224
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
    
    func unitTest() {
        let unitTest = PaddleMobileUnitTest.init(inDevice: MetalHelper.shared.device, inQueue: MetalHelper.shared.queue)
        unitTest.testConvAddBnRelu()
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        //        return
        let queue = MetalHelper.shared.queue
        
        textureLoader = MTKTextureLoader.init(device: MetalHelper.shared.device)
        guard let appleImage = UIImage.init(named: "banana.jpeg"), let cgImage = appleImage.cgImage else {
            fatalError(" image nil !")
        }
        
        let texture = try? textureLoader.newTexture(cgImage: cgImage, options: [:]) ?! " texture loader error"
        
        guard let inTexture = texture else {
            fatalError(" texture is nil !")
        }
        
        scaleTexture(queue: queue, input: inTexture) { (inputTexture) in
            do {
                try self.executor.predict(input: inputTexture, expect: [1, 224, 224, 3], completionHandle: { (result) in
                    print(result.resultArr.top(r: 5))
                }, preProcessKernle: self.preprocessKernel)
            } catch let error {
                print(error)
            }
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let queue = MetalHelper.shared.queue
        let loader = Loader<Float32>.init()
        preprocessKernel = PreProccess.init(device: MetalHelper.shared.device)

        do {
            let modelPath = Bundle.main.path(forResource: "model", ofType: nil) ?! "model null"
            let paraPath = Bundle.main.path(forResource: "params", ofType: nil) ?! "para null"
            program = try loader.load(device: MetalHelper.shared.device, modelPath: modelPath, paraPath: paraPath)
            executor = try Executor<Float32>.init(inDevice: MetalHelper.shared.device, inQueue: queue, inProgram: program)
        } catch let error {
            print(error)
        }
    }
}

