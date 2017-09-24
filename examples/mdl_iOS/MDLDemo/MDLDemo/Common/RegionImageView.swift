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

import UIKit


class RegionImageView: UIView {
    var resultRegion: [Float]{
        set{
            rectView.resultRegion = newValue
        }
        get{
            return rectView.resultRegion
        }
    }
    
    let imageView: UIImageView
    let rectView: RectView
    class RectView: UIView {
        var resultRegion: [Float] = []
    
        /// 不同模型框选使用的坐标信息 顺序是不一样的
        override func draw(_ rect: CGRect) {
            if resultRegion.count == 4{
                switch modelType {
                case .GoogleNet:
                    if let context = UIGraphicsGetCurrentContext(){
                        let widthScale: Float = Float(self.bounds.width/224.0);
                        let heightScale: Float = Float(self.bounds.height/224.0);
                        
                        context.stroke(CGRect.init(x: resultRegion[0] * widthScale, y: resultRegion[1] * heightScale, width: (resultRegion[2] - resultRegion[0]) * widthScale, height: (resultRegion[3] - resultRegion[1]) * heightScale))
                        
                        UIColor.red.setStroke()
                        context.strokePath()
                    }
                    
                    break
                case .MobileNet:
                    if let context = UIGraphicsGetCurrentContext(){
                        let widthScale: Float = Float(self.bounds.width/224.0);
                        let heightScale: Float = Float(self.bounds.height/224.0);
                        let result = resultRegion.map{$0 * 224}
                        context.stroke(CGRect.init(x: result[0] * widthScale, y: result[2] * heightScale, width: (result[1] - result[0]) * widthScale, height: (result[3] - result[2]) * heightScale))
                        
                        UIColor.red.setStroke()
                        context.strokePath()
                    }
                    
                    break
                }
            }
        }
    }
    
    override func setNeedsDisplay() {
        super.setNeedsDisplay()
        self.rectView.setNeedsDisplay()
    }
    override init(frame: CGRect) {
        imageView = UIImageView(frame: CGRect.init(x: 0, y: 0, width: frame.size.width, height: frame.size.height))
        rectView = RectView(frame: CGRect.init(x: 0, y: 0, width: frame.size.width, height: frame.size.height))
        super.init(frame: frame)
        self.addSubview(imageView)
        rectView.backgroundColor = UIColor.clear
        self.addSubview(rectView)
    }
    
    override var frame: CGRect{
        get{
            return super.frame
        }
        set{
            super.frame = newValue
            imageView.frame = CGRect.init(x: 0, y: 0, width: newValue.size.width, height: newValue.size.height)
        }
    }
    
    var image: UIImage?{
        get{
            return imageView.image
        }
        set{
            imageView.image = newValue
        }
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(_ rect: CGRect) {
    }
}
