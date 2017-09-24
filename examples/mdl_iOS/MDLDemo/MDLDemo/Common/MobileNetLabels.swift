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

import Foundation

/**
 The list of MobileNet label names, loaded from pre_words.txt.
 */
public class ImageNetLabels {
    private var labels = [String](repeating: "", count: 1000)
    
    public init() {
        if let path = Bundle.main.path(forResource: "pre_words", ofType: "txt") {
            for (i, line) in lines(filename: path).enumerated() {
                if i < 1000 {
                    labels[i] = line.substring(from: line.index(line.startIndex, offsetBy: 10))
                }
            }
        }
    }
    
    private func lines(filename: String) -> [String] {
        do {
            let text = try String(contentsOfFile: filename, encoding: .ascii)
            let lines = text.components(separatedBy: NSCharacterSet.newlines)
            return lines
        } catch {
            fatalError("Could not load file: \(filename)")
        }
    }
    
    public subscript(i: Int) -> String {
        return labels[i]
    }
}
