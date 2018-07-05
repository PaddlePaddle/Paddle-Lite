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

class ViewController: UIViewController {
    let device: MTLDevice! = MTLCreateSystemDefaultDevice()
    var textureLoader: MTKTextureLoader!
//    let queue: MTLCommandQueue
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let queue = device.makeCommandQueue()
        
        textureLoader = MTKTextureLoader.init(device: device)
        guard let appleImage = UIImage.init(named: "apple.jpg"), let cgImage = appleImage.cgImage else {
            fatalError(" image nil !")
        }
        
        let texture = try? textureLoader.newTexture(cgImage: cgImage, options: [:]) ?! " texture loader error"
        
        guard let inTexture = texture else {
            fatalError(" texture is nil !")
        }
    
        let loader = Loader<Float16>.init()
        do {
            let modelPath = Bundle.main.path(forResource: "model", ofType: nil) ?! "model null"
            let paraPath = Bundle.main.path(forResource: "params", ofType: nil) ?! "para null"
            let program = try loader.load(device: device, modelPath: modelPath, paraPath: paraPath)
            let executor = try Executor<Float16>.init(inDevice: device, inQueue: queue!, inProgram: program)
            let output = try executor.predict(input: inTexture, expect: [1, 227, 227, 3])
            print(output)
        } catch let error {
            print(error)
        }
    }

}

