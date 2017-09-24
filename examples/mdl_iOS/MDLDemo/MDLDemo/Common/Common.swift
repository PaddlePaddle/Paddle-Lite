//
//  Common.swift
//  MDLDemo
//
//  Created by liuRuiLong on 2017/9/13.
//  Copyright © 2017年 baidu. All rights reserved.
//

import Foundation


// 自定义 ?!  如果 ?! 前的返回值为一个可选值, 则进行隐式解包, 如果有值则返回这个值, 如果为nil 则fatalError 传入的信息
precedencegroup ExecutedOrFatalError{
    associativity: left
    higherThan: AssignmentPrecedence
}
infix operator ?!: ExecutedOrFatalError
func ?!<T>(option: T?, excuteOrError: @autoclosure () -> String) -> T{
    if let inOpt = option {
        return inOpt
    }else{
        fatalError(excuteOrError())
    }
}

extension Array where Element: Comparable{
    func top(r: Int) -> [(Int, Element)] {
        precondition(r <= self.count)
        return Array<(Int, Element)>(zip(0..<self.count, self).sorted{ $0.1 > $1.1 }.prefix(through: r - 1))
    }
}

