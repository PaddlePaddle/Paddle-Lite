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

import Foundation
import paddle_mobile

public class MobileNetCombined: Net {
    @objc public override init(device: MTLDevice) {
        super.init(device: device)
        except = 0
        modelPath = Bundle.main.path(forResource: "combined_mobilenet_model", ofType: nil) ?! "model null"
        paramPath = Bundle.main.path(forResource: "combined_mobilenet_params", ofType: nil) ?! "para null"
        inputDim = Dim.init(inDim: [1, 224, 224, 3])
        metalLoadMode = .LoadMetalInCustomMetalLib
        metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib")
        useMPS = true
    }
    let labels = PreWords.init(fileName: "vision_synset")

    class PreWords {
        var contents: [String] = []
        init(fileName: String, type: String = "txt", inBundle: Bundle = Bundle.main) {
            if let filePath = inBundle.path(forResource: fileName, ofType: type) {
                let string = try! String.init(contentsOfFile: filePath)
                contents = string.components(separatedBy: CharacterSet.newlines).filter{$0.count > 10}.map{
                    String($0[$0.index($0.startIndex, offsetBy: 10)...])
                }
            }else{
                fatalError("no file call \(fileName)")
            }
        }
        subscript(index: Int) -> String {
            return contents[index]
        }
    }
    
    override  public func resultStr(res: [ResultHolder]) -> String {
        let firstRes = res[0]
        let resPointer = firstRes.result
        var s: [String] = []
        (0..<firstRes.capacity).map { resPointer[$0] }.top(r: 5).enumerated().forEach{
            s.append(String(format: "%d: %@ (%3.2f%%)", $0 + 1, labels[$1.0], $1.1 * 100))
        }
        return s.joined(separator: "\n")
    }
    
}
