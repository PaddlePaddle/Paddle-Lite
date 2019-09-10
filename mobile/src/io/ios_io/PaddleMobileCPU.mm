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

#import "PaddleMobileCPU.h"
#import "framework/load_ops.h"
#import "framework/tensor.h"
#import "io/paddle_mobile.h"
#import <memory>
#import <vector>

@interface PaddleMobileCPUResult()

-(void)toSetOutput:(float *)output;

-(void)toSetOutputSize:(int)outputSize;

@end

@implementation PaddleMobileCPUResult

-(void)releaseOutput {
  delete [] _output;
  _output = nil;
  _outputSize = 0;
}

-(void)toSetOutput:(float *)output {
  _output = output;
}

-(void)toSetOutputSize:(int)outputSize {
  _outputSize = outputSize;
}

-(void)toSetDim:(NSArray <NSNumber *> *)dim {
  _dim = dim;
}

@end

@implementation  PaddleMobileCPUConfig

-(instancetype)init {
  if (self = [super init]) {
    self.threadNum = 1;
    self.optimize = YES;
  }
  return self;
}

@end

@interface  PaddleMobileCPU()
{
  paddle_mobile::PaddleMobile<paddle_mobile::CPU, float> *pam_;
  BOOL loaded_;
}

@property (strong, nonatomic) PaddleMobileCPUConfig *config;

@end

@implementation PaddleMobileCPU

static std::mutex shared_mutex;

- (instancetype)initWithConfig:(PaddleMobileCPUConfig *)config {
  if (self = [super init]) {
    paddle_mobile::PaddleMobileConfigInternal configInternal;
    configInternal.load_when_predict = config.loadWhenPredict;
    pam_ = new paddle_mobile::PaddleMobile<paddle_mobile::CPU, float>();
    _config = config;
  }
  return self;
}

-(instancetype)init {
  if (self = [super init]) {
    _config = [[PaddleMobileCPUConfig alloc] init];
    pam_ = new paddle_mobile::PaddleMobile<paddle_mobile::CPU, float>();
  }
  return self;
}

- (void)dealloc {
  if (pam_) {
    delete pam_;
    pam_ = nullptr;
  }
}

+ (instancetype)sharedInstance{
  static dispatch_once_t onceToken;
  static id sharedManager = nil;
  dispatch_once(&onceToken, ^{
    sharedManager = [[[self class] alloc] init];
  });
  return sharedManager;
}

- (BOOL)loadModel:(NSString *)modelPath andWeightsPath:(NSString *)weighsPath {
  std::lock_guard<std::mutex> lock(shared_mutex);
  std::string model_path_str = std::string([modelPath UTF8String]);
  std::string weights_path_str = std::string([weighsPath UTF8String]);
  pam_->SetThreadNum(self.config.threadNum);
  if (loaded_ = pam_->Load(model_path_str, weights_path_str, self.config.optimize, false, 1, self.config.loddable)) {
    return YES;
  } else {
    return NO;
  }
}

- (BOOL)LoadCombinedMemory:(size_t)modelLen
               andModelBuf:(const uint8_t *)modelBuf
         andModelParamsLen:(size_t)combinedParamsLen
      andCombinedParamsBuf:(const uint8_t *)combinedParamsBuf {
  std::lock_guard<std::mutex> lock(shared_mutex);
  pam_->SetThreadNum(self.config.threadNum);
  return loaded_ = pam_->LoadCombinedMemory(modelLen, modelBuf, combinedParamsLen,
          const_cast<uint8_t*>(combinedParamsBuf), self.config.optimize, false, 1, self.config.loddable);
}

- (BOOL)load:(NSString *)modelAndWeightPath{
  std::lock_guard<std::mutex> lock(shared_mutex);
  std::string model_path_str = std::string([modelAndWeightPath UTF8String]);
  if (loaded_ = pam_->Load(model_path_str, self.config.optimize, false, 1, self.config.loddable)) {
    return YES;
  } else {
    return NO;
  }
}


-(void)preprocess:(CGImageRef)image
           output:(float *)output
            means:(NSArray<NSNumber *> *)means
        scale:(float)scale
        dim:(NSArray<NSNumber *> *)dim {
  std::lock_guard<std::mutex> lock(shared_mutex);

  if (means == nil) {
    means = @[@0, @0, @0];
  }

  // dim to c++ vector, get numel
  std::vector<int64_t > dim_vec;
  int numel = 1;
  for (int k = 0; k < dim.count; ++k) {
    int d = dim[k].intValue;
    numel *= d;
    dim_vec.push_back(d);
  }

  const int sourceRowBytes = CGImageGetBytesPerRow(image);
  const int imageWidth = CGImageGetWidth(image);
  const int imageHeight = CGImageGetHeight(image);
  const int imageChannels = 4;
  CGDataProviderRef provider = CGImageGetDataProvider(image);
  CFDataRef cfData = CGDataProviderCopyData(provider);
  const UInt8 *input = CFDataGetBytePtr(cfData);

  int wanted_input_width = dim_vec[3];
  int wanted_input_height = dim_vec[2];
  int wanted_input_channels = dim_vec[1];

  for (int c = 0; c < wanted_input_channels; ++c) {
    float *out_channel = output + c * wanted_input_height * wanted_input_width;
    for (int y = 0; y < wanted_input_height; ++y) {
      float *out_row = out_channel + y * wanted_input_width;
      for (int x = 0; x < wanted_input_width; ++x) {
        int in_row = (y * imageHeight) / wanted_input_height;
        int in_col = (x * imageWidth) / wanted_input_width;
        const UInt8 *in_pixel = input + (in_row * sourceRowBytes) + (in_col * imageChannels);
        float *out_pos = out_row + x;
        *out_pos = (in_pixel[2 - c] - means[c].floatValue) * scale;
      }
    }
  }

}

-(void)preprocess:(const UInt8 *)input output:(float *)output bytesPerRow:(int)bytesPerRow imageWidth:(int)imageWidth imageHeight:(int)imageHeight imageChannels:(int)imageChannels means:(NSArray<NSNumber *> *)means stds:(NSArray<NSNumber *> *)stds scale:(float)scale dim:(std::vector<int64_t>)dim {
  if (means == nil) {
    means = @[@0, @0, @0];
  }
  if (stds == nil) {
    stds = @[@1, @1, @1];
  }

  int wanted_input_width = dim[3];
  int wanted_input_height = dim[2];
  int wanted_input_channels = dim[1];

  for (int c = 0; c < wanted_input_channels; ++c) {
    float *out_channel = output + c * wanted_input_height * wanted_input_width;
    for (int y = 0; y < wanted_input_height; ++y) {
      float *out_row = out_channel + y * wanted_input_width;
      for (int x = 0; x < wanted_input_width; ++x) {
        int in_row = (y * imageHeight) / wanted_input_height;
        int in_col = (x * imageWidth) / wanted_input_width;
        const UInt8 *in_pixel = input + (in_row * bytesPerRow) + (in_col * imageChannels);
        float *out_pos = out_row + x;
        *out_pos = (in_pixel[2 - c] - means[c].floatValue) / stds[c].floatValue * scale;
      }
    }
  }
}

- (PaddleMobileCPUResult *)predictInput:(float *)input
                      dim:(NSArray<NSNumber *> *)dim {
  std::lock_guard<std::mutex> lock(shared_mutex);
  if (!loaded_) {
    printf("PaddleMobile doesn't be loaded yet");
    return nil;
  }

  if (dim.count != 4) {
    printf("dim must have 4 elements");
    return nil;
  }

  // dim to c++ vector, get numel
  std::vector<int64_t > dim_vec;
  int numel = 1;
  for (int k = 0; k < dim.count; ++k) {
    int d = dim[k].intValue;
    numel *= d;
    dim_vec.push_back(d);
  }

  paddle_mobile::framework::Tensor input_tensor;
  paddle_mobile::framework::DDim dims = paddle_mobile::framework::make_ddim(dim_vec);
  float *input_ptr = input_tensor.mutable_data<float>(dims);
  memcpy(input_ptr, input,
         numel * sizeof(float));

  pam_->Predict(input_tensor);
  std::shared_ptr<paddle_mobile::framework::Tensor> output = pam_->Fetch();

  auto output_dims = output->dims();
  std::vector<int64_t> output_dim_vec = vectorize(output_dims);
  NSMutableArray <NSNumber *> *ocDim = [NSMutableArray array];
  for (int i = 0; i < output_dim_vec.size(); ++i) {
    NSNumber *num = [NSNumber numberWithLongLong:output_dim_vec[i]];
    [ocDim addObject:num];
  }

  float *output_pointer = new float[output->numel()];

  memcpy(output_pointer, output->data<float>(),
         output->numel() * sizeof(float));

  PaddleMobileCPUResult *cpuResult = [[PaddleMobileCPUResult alloc] init];
  [cpuResult toSetOutput: output_pointer];
  [cpuResult toSetDim: ocDim];
  [cpuResult toSetOutputSize: output->numel()];

  return cpuResult;
}

- (PaddleMobileCPUResult *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim means:(NSArray<NSNumber *> *)means stds:(NSArray<NSNumber *> *)stds scale:(float)scale {
  std::lock_guard<std::mutex> lock(shared_mutex);
  if (!loaded_) {
    printf("PaddleMobile doesn't be loaded yet");
    return nil;
  }

  if (dim.count != 4) {
    printf("dim must have 4 elements");
    return nil;
  }

  // dim to c++ vector, get numel
  std::vector<int64_t > dim_vec;
  int numel = 1;
  for (int k = 0; k < dim.count; ++k) {
    int d = dim[k].intValue;
    numel *= d;
    dim_vec.push_back(d);
  }

  const int sourceRowBytes = CGImageGetBytesPerRow(image);
  const int image_width = CGImageGetWidth(image);
  const int image_height = CGImageGetHeight(image);
  const int image_channels = 4;
  CGDataProviderRef provider = CGImageGetDataProvider(image);
  CFDataRef cfData = CGDataProviderCopyData(provider);
  const UInt8 *input = CFDataGetBytePtr(cfData);

  // sample image
  float *output = (float *)malloc(numel*sizeof(float));
  [self preprocess:input output:output bytesPerRow:sourceRowBytes imageWidth:image_width imageHeight:image_height imageChannels:image_channels means:means stds:stds scale:scale dim:dim_vec];
  float *dataPointer = nullptr;
  if (nullptr != output) {
    dataPointer = output;
  } else {
    return nil;
  }

  paddle_mobile::framework::Tensor input_tensor;
  paddle_mobile::framework::DDim dims = paddle_mobile::framework::make_ddim(dim_vec);
  float *input_ptr = input_tensor.mutable_data<float>(dims);
  memcpy(input_ptr, dataPointer,
         numel * sizeof(float));

  pam_->Predict(input_tensor);
  std::shared_ptr<paddle_mobile::framework::Tensor> output_tensor = pam_->Fetch();

  auto output_dims = output_tensor->dims();
  std::vector<int64_t> output_dim_vec = vectorize(output_dims);
  NSMutableArray <NSNumber *> *ocDim = [NSMutableArray array];
  for (int i = 0; i < output_dim_vec.size(); ++i) {
    NSNumber *num = [NSNumber numberWithLongLong:output_dim_vec[i]];
    [ocDim addObject:num];
  }

  float *output_pointer = new float[output_tensor->numel()];
  memcpy(output_pointer, output_tensor->data<float>(),
         output_tensor->numel() * sizeof(float));
  PaddleMobileCPUResult *cpuResult = [[PaddleMobileCPUResult alloc] init];
  [cpuResult toSetOutput: output_pointer];
  [cpuResult toSetDim: ocDim];
  [cpuResult toSetOutputSize: output_tensor->numel()];

  free(output);
  CFRelease(cfData);
  cfData = NULL;

  return cpuResult;
}

- (PaddleMobileCPUResult *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim {
  return [self predict:image dim:dim means:nil stds:nil scale:1];
}

- (PaddleMobileCPUResult *)predict:(CGImageRef)image dim:(NSArray<NSNumber *> *)dim means:(NSArray<NSNumber *> *)means scale:(float)scale {
  return [self predict:image dim:dim means:means stds:nil scale:scale];
}

- (PaddleMobileCPUResult *)fetchOutput{
  if (pam_ && loaded_) {
    auto tensorPtr = pam_->Fetch();
    float *output_pointer = new float[tensorPtr->numel()];
    memcpy(output_pointer, tensorPtr->data<float>(),
           tensorPtr->numel() * sizeof(float));
    auto dims = tensorPtr->dims();
    std::vector<int64_t> dim_vec = vectorize(dims);


    NSMutableArray <NSNumber *> *ocDim = [NSMutableArray array];
    for (int i = 0; i < dim_vec.size(); ++i) {
      NSNumber *num = [NSNumber numberWithLongLong:dim_vec[i]];
      [ocDim addObject:num];
    }

    PaddleMobileCPUResult *cpuResult = [[PaddleMobileCPUResult alloc] init];
    [cpuResult toSetOutput: output_pointer];
    [cpuResult toSetDim: ocDim];
    [cpuResult toSetOutputSize: tensorPtr->numel()];

    return cpuResult;
  }
  return nil;
}

- (PaddleMobileCPUResult *)fetchOutputWithKey:(NSString *)key{
  if (pam_ && loaded_ && key.length) {
    auto tensorPtr = pam_->Fetch(std::string([key cStringUsingEncoding:NSUTF8StringEncoding]));
    float *output_pointer = new float[tensorPtr->numel()];
    memcpy(output_pointer, tensorPtr->data<float>(),
           tensorPtr->numel() * sizeof(float));

    auto dims = tensorPtr->dims();
    std::vector<int64_t> dim_vec = vectorize(dims);

    NSMutableArray <NSNumber *> *ocDim = [NSMutableArray array];
    for (int i = 0; i < dim_vec.size(); ++i) {
      NSNumber *num = [NSNumber numberWithLongLong:dim_vec[i]];
      [ocDim addObject:num];
    }

    PaddleMobileCPUResult *cpuResult = [[PaddleMobileCPUResult alloc] init];
    [cpuResult toSetOutput: output_pointer];
    [cpuResult toSetDim: ocDim];
    [cpuResult toSetOutputSize: tensorPtr->numel()];

    return cpuResult;
  }
  return nil;
}

- (void)clear{
  std::lock_guard<std::mutex> lock(shared_mutex);
  if (pam_) {
    pam_->Clear();
  }
}

@end
