#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "lite/core/tensor.h"

typedef enum {
  /*! \brief signed integer */
  kPTInt = 0U,
  /*! \brief unsigned integer */
  kPTUInt = 1U,
  /*! \brief IEEE floating point */
  kPTFloat = 2U,
  /*!
   * \brief Opaque handle type, reserved for testing purposes.
   * Frameworks need to agree on the handle data type for the exchange to be well-defined.
   */
  kPTOpaqueHandle = 3U,
  /*! \brief bfloat16 */
  kPTBfloat = 4U,
  /*! \brief int8 */
  kPTInt8 = 5U,
  /*!
   * \brief complex number
   * (C/C++/Python layout: compact struct per complex number)
   */
  kPTComplex = 6U,
} PT_DataType;

typedef struct {
  /* pointer to data in bytes and byte_size */
  void *bytes;
  size_t byte_size;
  /* pointer to data of shape and dim_size */
  uint64_t *dims;
  size_t dim_size;
  size_t byte_offset;
  PT_DataType data_type;
} PortableTensor;

typedef enum {
  Softmax = 1U,
  Fc = 2U,
} SupportedOp;

typedef struct {
  PortableTensor x;
  PortableTensor output;
  int axis; 
  bool use_cudnn;
} PT_SoftmaxParam;

typedef struct {
  PortableTensor input;
  PortableTensor w;
  PortableTensor bias;
  PortableTensor output;
  PortableTensor scale;
  float input_scale;
  float output_scale;
  bool flag_act;
  bool flag_trans_weights;
} PT_FcParam;

/* helper functions for converting Lite::Tensor to PortableTensor */
PortableTensor convert_to_portable_tensor(paddle::lite::Tensor* lite_tensor, PT_DataType dtype, bool is_mut);
PortableTensor convert_to_portable_tensor(void* data, PT_DataType dtype, const paddle::lite::DDimLite *dims);
void check_interactive();

/* CA APIs */
typedef uint64_t handle_t;
extern "C" {
  // Common APIs
  handle_t create_tee_param(SupportedOp op, void* pt_param_ptr);
  int tee_run(SupportedOp op, handle_t param_handle);
  int fetch_output_tensor(SupportedOp op, handle_t param_handle, PortableTensor output_tensor);
}
