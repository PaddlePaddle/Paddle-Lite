//
//  ViewController.swift
//  paddle-mobile-unit-test
//
//  Created by liuRuiLong on 2018/8/10.
//  Copyright © 2018年 orange. All rights reserved.
//

import UIKit
import Metal
//import MetalKit
import paddle_mobile

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        let device = Metal.MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let test = PaddleMobileUnitTest.init(
            inDevice: device,
            inQueue: queue
        )
        test.testConcat()
//        test.testReshape()
//        test.testTranspose()
        print(" done ")
    }

}
