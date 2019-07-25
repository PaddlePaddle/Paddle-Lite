//
//  ViewController.swift
//  MobileNetDemo
//
//  Created by liuRuiLong on 2019/1/4.
//  Copyright © 2019 Ray. All rights reserved.
//

import UIKit
import paddle_mobile

class ViewController: UIViewController {
    @IBOutlet weak var resultTextView: UITextView!
    @IBOutlet weak var selectImageView: UIImageView!
    @IBOutlet weak var elapsedTimeLabel: UILabel!
    var net: MobileNet!
    var runner: Runner!
    var toPredictTexture: MTLTexture?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        GlobalConfig.shared.computePrecision = .Float16
        net = MobileNet.init(device: MetalHelper.shared.device)
        runner = Runner.init(inNet: net, commandQueue: MetalHelper.shared.queue)
        
        if let selectImage = UIImage.init(named: "banana.jpeg") {
            selectImageView.image = selectImage
            runner.getTexture(image: selectImage.cgImage!) {[weak self] (texture) in
                self?.toPredictTexture = texture
            }
        }
        
    }
    
    @IBAction func loadAct(_ sender: Any) {
        if runner.load() {
            let resutText = " load success ! "
            print(resutText)
            self.resultTextView.text = resutText
        } else {
            print("load fail!!!")
        }
    }
    
    @IBAction func selectImageAct(_ sender: Any) {
        let imagePicker = UIImagePickerController()
        imagePicker.sourceType = .camera
        imagePicker.delegate = self
        self.present(imagePicker, animated: true, completion: nil)
    }
    
    @IBAction func clearAct(_ sender: Any) {
        runner.clear()
    }
    
    @IBAction func predictAct(_ sender: Any) {
        
        if let texture = toPredictTexture {
            let beginDate = Date.init()
            runner.predict(texture: texture) { [weak self] (success, resultHolder) in
                if success, let inResultHolder = resultHolder {
                    let timeUse = Date.init().timeIntervalSince(beginDate)
                    DispatchQueue.main.async {
                        self?.elapsedTimeLabel.text = "\(timeUse * 1000)ms"
                        self?.resultTextView.text = self?.net.resultStr(res: inResultHolder)
                    }
                    
                } else {
                    print(" predict fail ")
                }
            }
        } else {
            print(" toPredictTexture is nil ")
        }
        
    }
    
}

extension ViewController:  UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true){[weak self] in
            guard let sSelf = self, let image =  info["UIImagePickerControllerOriginalImage"] as? UIImage else {
                print("no image!!!")
                return
            }
            sSelf.selectImageView.image = image
            sSelf.runner.getTexture(image: image.cgImage!, getTexture: { (texture) in
                sSelf.toPredictTexture = texture
            })
        }
    }
}

