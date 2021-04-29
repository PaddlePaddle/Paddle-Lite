//
//  mps_conv_datasource.m
//  PaddleLiteiOS
//
//  Created by hxwc on 2021/3/30.
//

#include "lite/backends/metal/mps_conv_datasource.h"

@implementation MPSConvDataSource

- (MPSDataType)dataType {
	return MPSDataTypeFloat16;
}

- (MPSCNNConvolutionDescriptor * __nonnull) descriptor {
	return _descriptor;
}

- (void *)weights {
	return _weights;
}

- (float *)biasTerms {
	return _biasTerms;
}

- (BOOL)load {
	return YES;
}

- (void)purge{

}

- (NSString*)label {
	return @"mps_conv_add_relu_label";
}

- (nonnull id)copyWithZone:(nullable NSZone *)zone {
	return self;
}

@end
