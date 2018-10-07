//
//  Tools.swift
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/7/26.
//  Copyright © 2018年 orange. All rights reserved.
//

import Foundation

func writeToLibrary<P: PrecisionType>(fileName: String, array: [P]) {
    let libraryPath = NSSearchPathForDirectoriesInDomains(.libraryDirectory, .userDomainMask, true).last ?! " library path get error "
    let filePath = libraryPath + "/" + fileName
    let fileManager = FileManager.init()
    fileManager.createFile(atPath: filePath, contents: nil, attributes: nil)
    let fileHandler = FileHandle.init(forWritingAtPath: filePath) ?! " file handler nil "
    let data = Data.init(buffer: UnsafeBufferPointer.init(start: array, count: array.count))
    fileHandler.write(data)
    fileHandler.closeFile()
}

