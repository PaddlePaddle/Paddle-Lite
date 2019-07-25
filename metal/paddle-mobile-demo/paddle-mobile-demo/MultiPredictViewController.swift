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

class MultiPredictViewController: UIViewController {
    var runner1: Runner!
    var runner2: Runner!
    override func viewDidLoad() {
        super.viewDidLoad()
        let mobileNet = try! MobileNet_ssd_hand.init(device: MetalHelper.shared.device)
        let genet = try! Genet.init(device: MetalHelper.shared.device)
        runner1 = try! Runner.init(inNet: mobileNet, commandQueue: MetalHelper.shared.queue)
        let queue2 = MetalHelper.shared.device.makeCommandQueue()
        
        runner2 = try! Runner.init(inNet: genet, commandQueue: MetalHelper.shared.queue)
    }
    
    @IBAction func predictAct(_ sender: Any) {
        let success = self.runner2.load()
        //    DispatchQueue.global().async {
        let image1 = UIImage.init(named: "hand.jpg")
        //      let success = self.runner2.load()
        //      if success {
        //        for i in 0..<10000 {
        //          print(i)
        //          self.runner2.predict(cgImage: image1!.cgImage!, completion: { (success, res) in
        //            print("result1: ")
        ////            print(res)
        //          })
        //        }
        //      } else {
        //        print("load failed")
        //      }
        //      self.runner1.clear()
        //    }
        //    return
        //    DispatchQueue.global().async {
        ////      sleep(1)
        //      let image1 = UIImage.init(named: "banana.jpeg")
        ////      if success {
        //        for _ in 0..<10 {
        //          self.runner2.predict(cgImage: image1!.cgImage!, completion: { (success, res) in
        //            print("result2: ")
        //            print(res)
        //          })
        //        }
        ////      } else {
        ////        print("load failed")
        ////      }
        ////      self.runner2.clear()
        //    }
    }
}
