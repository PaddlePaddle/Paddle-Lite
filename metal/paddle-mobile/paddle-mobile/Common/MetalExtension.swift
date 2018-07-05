//
//  MetalExtension.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/2.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

fileprivate var defaultMetalLibrary: MTLLibrary?
fileprivate var paddleMobileMetalLibrary: MTLLibrary?

extension MTLDevice {
    func defaultLibrary() -> MTLLibrary {
        if defaultMetalLibrary == nil {
            defaultMetalLibrary = makeDefaultLibrary()
        }
        if let inDefaultLib = defaultMetalLibrary {
            return inDefaultLib
        } else {
            fatalError(" default metal libary is nil")
        }
    }
    
    func paddleMobileLibrary() -> MTLLibrary {
        if paddleMobileMetalLibrary == nil {
            guard let path = Bundle.init(for: Kernel.self).path(forResource: "default", ofType: "metallib") else {
                fatalError("Counld't find paddle mobile library")
            }
            do {
                print(path)
                paddleMobileMetalLibrary = try makeLibrary(filepath: path)
            } catch _ {
                fatalError("Counld't load paddle mobile library")
            }
        }
        
        if let inPaddleMobileLib = paddleMobileMetalLibrary {
            return inPaddleMobileLib
        } else {
            fatalError("PaddleMobile metal libary is nil")
        }
    }
    
    
    func pipeLine(funcName: String, inPaddleMobileLib: Bool = true) -> MTLComputePipelineState {
        let useLib = inPaddleMobileLib ? paddleMobileLibrary() : defaultLibrary()
        guard let function = useLib.makeFunction(name: funcName) else {
            fatalError(" function " + funcName + " not found")
        }
        do {
            let pipLine = try makeComputePipelineState(function: function)
            return pipLine
        } catch _ {
            fatalError("make pip line error occured")
        }
        
    }
}

extension MTLComputeCommandEncoder {
    func dispatch(computePipline: MTLComputePipelineState, outTexture: MTLTexture) {
        let slices = (outTexture.depth + 3)/4
        
        let width = computePipline.threadExecutionWidth
        let height = computePipline.maxTotalThreadsPerThreadgroup/width
        let threadsPerGroup = MTLSize.init(width: width, height: height, depth: 1)
    
        print(" threads per group: \(threadsPerGroup) ")
        
        print(" out texture width: \(outTexture.width) , out texture height: \(outTexture.height)")
        
        let groupWidth = (outTexture.width + width - 1)/width
        let groupHeight = (outTexture.height + height - 1)/height
        let groupDepth = slices
        let groups = MTLSize.init(width: groupWidth, height: groupHeight, depth: groupDepth)
        
        print("groups: \(groups) ")
        
        setComputePipelineState(computePipline)
        dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    }
}










