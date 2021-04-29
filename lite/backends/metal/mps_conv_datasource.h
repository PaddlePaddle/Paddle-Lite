//
//  mps_conv_datasource.h
//  PaddleLiteiOS
//
//  Created by hxwc on 2021/3/30.
//

#ifndef MPS_CONV_DATASOURCE_H
#define MPS_CONV_DATASOURCE_H

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSConvDataSource : NSObject <MPSCNNConvolutionDataSource> {

}
@property (nonatomic, assign)void* weights;
@property (nonatomic, assign)float* biasTerms;
@property (nonatomic, strong)MPSCNNConvolutionDescriptor* descriptor;

@end

#endif
