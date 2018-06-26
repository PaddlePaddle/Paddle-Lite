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
import paddle_mobile

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        let loader = Loader<Float>.init()
        do {
            let modelPath = Bundle.main.path(forResource: "model", ofType: nil) ?! "model null"
            let paraPath = Bundle.main.path(forResource: "params", ofType: nil) ?! "para null"
            let program = try loader.load(modelPath: modelPath, paraPath: paraPath)
            let executor = try Executor<Float>.init(program: program)
            executor.predict()
        } catch let error {
            print(error)
        }
    }

}

