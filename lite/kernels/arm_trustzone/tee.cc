#include "lite/api/paddle_place.h"
#include "lite/core/tensor.h"
#include "lite/kernels/arm_trustzone/tee.h"
#include <iomanip>

PortableTensor convert_to_portable_tensor(paddle::lite::Tensor* lite_tensor, PT_DataType dtype, bool is_mut) {
  PortableTensor* portable_tensor = new PortableTensor();
  // set bytes
  if (is_mut) {
    if (dtype == PT_DataType::kPTFloat)
      portable_tensor->bytes = (void*)lite_tensor->mutable_data<float>();
    else if (dtype == PT_DataType::kPTInt8)
      portable_tensor->bytes = (void*)lite_tensor->mutable_data<int8_t>();
  }
  else {
    portable_tensor->bytes = lite_tensor->raw_data();
  }

  // set dims and dim_size
  portable_tensor->dim_size = (size_t)lite_tensor->dims().size();
  portable_tensor->dims = new uint64_t[portable_tensor->dim_size];
  size_t data_count = 1;
  std::vector<int64_t> dim_vec = lite_tensor->dims().Vectorize();
  for (int i = 0; i < portable_tensor->dim_size; i++) {
    portable_tensor->dims[i] = dim_vec[i];
    data_count *= portable_tensor->dims[i];
  }
  CHECK_EQ(lite_tensor->numel(), data_count)
    << "The product of elements in shape[] is not equal to tensor.numel()";

  // set byte_size
  int bits = 8;
  switch (dtype) {
    case PT_DataType::kPTInt:
    case PT_DataType::kPTUInt:
    case PT_DataType::kPTFloat:
      bits = 32;
      break;
    case PT_DataType::kPTBfloat:
      bits = 16;
      break;
    case PT_DataType::kPTInt8:
      bits = 8;
      break;
  }
  portable_tensor->byte_size = data_count * bits/8;

  // set byte_offset
  portable_tensor->byte_offset = lite_tensor->offset();
  // set data_type
  portable_tensor->data_type = dtype;

  return *portable_tensor;
}

PortableTensor convert_to_portable_tensor(void* data, PT_DataType dtype, const paddle::lite::DDimLite *dims) {
  PortableTensor* portable_tensor = new PortableTensor();
  portable_tensor->bytes = data;
  portable_tensor->dim_size = (size_t)dims->size();
  portable_tensor->dims = new uint64_t[portable_tensor->dim_size];
  size_t data_count = 1;
  std::vector<int64_t> dim_vec = dims->Vectorize();
  for (int i = 0; i < portable_tensor->dim_size; i++) {
    portable_tensor->dims[i] = dim_vec[i];
    data_count *= portable_tensor->dims[i];
  }

  // set byte_size
  int bits = 8;
  switch (dtype) {
    case PT_DataType::kPTInt:
    case PT_DataType::kPTUInt:
    case PT_DataType::kPTFloat:
      bits = 32;
      break;
    case PT_DataType::kPTBfloat:
      bits = 16;
      break;
    case PT_DataType::kPTInt8:
      bits = 8;
      break;
  }

  portable_tensor->byte_size = data_count * bits/8;

  // set byte_offset
  portable_tensor->byte_offset = 0;
  // set data_type
  portable_tensor->data_type = dtype;

  return *portable_tensor;
}

void check_interactive(){
  char *interactive = getenv("INTERACTIVE");
  if (*interactive == '1') {
    std::cin.get();
  }
}
