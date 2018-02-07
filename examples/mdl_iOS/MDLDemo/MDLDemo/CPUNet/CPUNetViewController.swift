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

import MDL
import UIKit

enum ModelType {
    case GoogleNet, MobileNet
}

let modelType = ModelType.GoogleNet

class CPUNetViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    var imageView: RegionImageView?
    var loaded: Bool = false
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView = RegionImageView(frame: CGRect.init(x: 0, y: 20, width: view.bounds.size.width, height: view.bounds.height - 100))
        view.insertSubview(imageView!, at: 0)
        
        DispatchQueue.global().async {
            var modelPath: String?
            var weightPath: String?
            if modelType == .GoogleNet{
                MDLCPUNet.share.setThreadNum(number: 3)
                modelPath = Bundle.main.path(forResource: "g_model.min", ofType: "json")
                weightPath = Bundle.main.path(forResource: "g_data.min", ofType: "bin")
            }else if modelType == .MobileNet{
                MDLCPUNet.share.setThreadNum(number: 1)
                modelPath = Bundle.main.path(forResource: "m_model.min", ofType: "json")
                weightPath = Bundle.main.path(forResource: "m_data.min", ofType: "bin")
            }
           
            if !MDLCPUNet.share.load(modelPath: modelPath!, weightPath: weightPath!){
                fatalError(" load error")
            }else{
                self.loaded = true
            }
        }
    }
    
    @IBAction func takePictureAct(_ sender: Any) {
        let imagePicker = UIImagePickerController()
        imagePicker.sourceType = .camera
        imagePicker.delegate = self
        self.present(imagePicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true){[weak self] in
            guard let sSelf = self, let image = sSelf.eraseDirectionInfo(originImage: info["UIImagePickerControllerOriginalImage"] as? UIImage) else{
                fatalError("no image")
            }
            
            sSelf.imageView?.bounds.size.height = image.size.height/image.size.width * sSelf.view.bounds.width
            sSelf.imageView?.image = image
            
            if modelType == .GoogleNet{
                MDLCPUNet.share.predict(image: image.cgImage!, means: [148.0, 148.0, 148.0], completion: { (result) in
                    print("elapsed: \(result.elapsedTime) -- result: \(result.result)")
                    sSelf.imageView?.resultRegion = result.result
                    sSelf.imageView?.setNeedsDisplay()
                })
            }else if modelType == .MobileNet{
                MDLCPUNet.share.predict(image: image.cgImage!, means: [123.68, 116.78, 103.94], scale: 0.017, completion: { (result) in
                    print("elapsed: \(result.elapsedTime) -- result: \(result.result)")
                    sSelf.imageView?.resultRegion = result.result
                    sSelf.imageView?.setNeedsDisplay()
                })
            }
        }
    }
    
    func eraseDirectionInfo(originImage: UIImage?) -> UIImage? {
        guard let inImage = originImage else {
            return nil
        }
        UIGraphicsBeginImageContext(inImage.size)
        inImage.draw(in: CGRect.init(x: 0, y: 0, width: inImage.size.width, height: inImage.size.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage
    }
    
}

extension CGRect{
    init(x: Float, y: Float, width: Float, height: Float) {
        self.init(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width), height: CGFloat(height))
    }
}
