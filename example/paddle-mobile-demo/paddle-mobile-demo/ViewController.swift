//
//  ViewController.swift
//  paddle-mobile-demo
//
//  Created by liuRuiLong on 2018/6/20.
//  Copyright © 2018年 orange. All rights reserved.
//

import UIKit
import paddle_mobile

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        var date = Date.init()
        if let modelPath = Bundle.main.path(forResource: "model", ofType: nil), let paraPath = Bundle.main.path(forResource: "params", ofType: nil) {
            print(" bundlepath: " + modelPath)
            let loader = Loader<Float32>.init()
            try! loader.load(modelPath: modelPath, paraPath: paraPath)
        }
        let els = Date.init().timeIntervalSince(date)
        print(els)
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

