//
//  Multi-Predict-ViewController.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/9/14.
//  Copyright © 2018年 orange. All rights reserved.
//

import UIKit
import paddle_mobile

class MultiPredictViewController: UIViewController {
  var runner1: Runner!
  var runner2: Runner!
  override func viewDidLoad() {
    super.viewDidLoad()
    let mobileNet = MobileNet_ssd_hand.init(device: MetalHelper.shared.device)
    let genet = Genet.init(device: MetalHelper.shared.device)
    runner1 = Runner.init(inNet: mobileNet, commandQueue: MetalHelper.shared.queue, inPlatform: .GPU)
    let queue2 = MetalHelper.shared.device.makeCommandQueue()
    
    runner2 = Runner.init(inNet: genet, commandQueue: MetalHelper.shared.queue, inPlatform: .GPU)
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
