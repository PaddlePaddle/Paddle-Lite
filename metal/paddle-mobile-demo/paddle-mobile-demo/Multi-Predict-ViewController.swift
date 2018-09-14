//
//  Multi-Predict-ViewController.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/9/14.
//  Copyright © 2018年 orange. All rights reserved.
//

import UIKit
import paddle_mobile

class Multi_Predict_ViewController: UIViewController {
  var runner1: Runner!
  var runner2: Runner!
  override func viewDidLoad() {
    super.viewDidLoad()
//    let net = MobileNet_ssd_hand.init(device: MetalHelper.shared.device)
//    runner1 = Runner.init(inNet: <#T##Net#>, commandQueue: <#T##MTLCommandQueue?#>, inPlatform: <#T##Platform#>)
  }

  @IBAction func predictAct(_ sender: Any) {
    
  }
}
