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

#import "PaddleMobile.h"
#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    PaddleMobile *pam = [[PaddleMobile alloc] init];
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model" ofType:nil];
    NSString *paramPath = [[NSBundle mainBundle] pathForResource:@"params" ofType:nil];
    if (modelPath.length == 0 || paramPath.length == 0) {
        NSLog(@" need model and param");
        return;
    }
    
    if ([pam load:modelPath andWeightsPath:paramPath]) {
        NSLog(@"load success");
        UIImage *inputImage = [UIImage imageNamed:@"apple.jpg"];
        if (!inputImage) {
            NSLog(@" input image is nil");
            return;
        }
        
        NSDate *beforeDate = [NSDate date];
        NSArray *res = [pam predict:inputImage.CGImage dim:@[@1, @3, @224, @224] means:@[@148, @148, @148] scale:1.0];
        NSLog(@"res: %@", res);
        NSLog(@"elapsed time: %f", [[NSDate date] timeIntervalSinceDate:beforeDate]);
    }
}

@end
