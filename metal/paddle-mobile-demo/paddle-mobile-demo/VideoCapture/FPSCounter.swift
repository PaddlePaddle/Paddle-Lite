

import Foundation
import QuartzCore

public class FPSCounter {
    private(set) public var fps: Double = 0
    
    var frames = 0
    var startTime: CFTimeInterval = 0
    
    public func start() {
        frames = 0
        startTime = CACurrentMediaTime()
    }
    
    public func frameCompleted() {
        frames += 1
        let now = CACurrentMediaTime()
        let elapsed = now - startTime
        if elapsed > 0.1 {
            let current = Double(frames) / elapsed
            let smoothing = 0.75
            fps = smoothing*fps + (1 - smoothing)*current
            if elapsed > 1 {
                frames = 0
                startTime = CACurrentMediaTime()
            }
        }
    }
}
