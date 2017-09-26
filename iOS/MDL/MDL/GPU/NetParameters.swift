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

public enum NetParameterType {
    case weights
    case biases
}

/// 存储着网络层运算所需的 权重 或者 偏置
///store the weight or biase of the net layer
public protocol NetParameterData {
    var pointer: UnsafeMutablePointer<Float> { get }
}

public class ParameterData: NetParameterData{
    public var pointer: UnsafeMutablePointer<Float>
    let size: Int
    init(size: Int) {
        self.size = size
        pointer = UnsafeMutablePointer<Float>.allocate(capacity: size)
        pointer.initialize(to: 0.0)
    }
    deinit {
        pointer.deinitialize()
        pointer.deallocate(capacity: size)
    }
}

public class NetParameterLoaderBundle: NetParameterData {
    private var fileSize: Int
    private var fd: CInt!
    private var hdr: UnsafeMutableRawPointer!
    private(set) public var pointer: UnsafeMutablePointer<Float>
    
    public init?(name: String, count: Int, ext: String, bundle: Bundle = Bundle.main) {
        fileSize = count * MemoryLayout<Float>.stride
        guard let path = bundle.path(forResource: name, ofType: ext) else {
            print("Parameter Get Error: resource \"\(name)\" not found")
            return nil
        }
        
        fd = open(path, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        if fd == -1 {
            print("Parameter Get Error: failed to open \"\(path)\", error = \(errno)")
            return nil
        }
        
        hdr = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
        if hdr == nil {
            print("Parameter Get Error: mmap failed, errno = \(errno)")
            return nil
        }
        
        pointer = hdr.bindMemory(to: Float.self, capacity: count)
        if pointer == UnsafeMutablePointer<Float>(bitPattern: -1) {
            print("Parameter Get Error: mmap failed, errno = \(errno)")
            return nil
        }
    }
    
    deinit {
        if let hdr = hdr {
            let result = munmap(hdr, Int(fileSize))
            assert(result == 0, "Parameter Get Error: munmap failed, errno = \(errno)")
        }
        if let fd = fd {
            close(fd)
        }
    }
}



