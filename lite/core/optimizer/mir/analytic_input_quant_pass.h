// Analytic feed quant: bake host integer feed + in-graph calib dequant.
// Default: kInt8 XOR-centered (q = (int8_t)(b ^ 128)) + int8_to_fp*.
// Optional: kUInt8 raw greyscale + uint8_to_fp* (out = (u - 128) * scale),
// same math as int8 xor without host XOR buffer rewrite.
// Normalisation is explicit: x = (u - mean) / std in pixel units, set from
// opt --analytic_input_mean / --analytic_input_std (no guessing). The calib
// kernels centre at 128; mean != 128 is folded into the first conv's bias.
#pragma once
#include <limits>
#include <memory>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

class AnalyticInputQuantPass : public ProgramPass {
 public:
  // "int8" (default) or "uint8"
  void SetInputDtype(const std::string& dtype) { input_dtype_ = dtype; }
  const std::string& input_dtype() const { return input_dtype_; }
  // Pixel units (0..255). Both must be set; NaN = unset -> pass aborts.
  void SetInputMean(float mean) { input_mean_ = mean; }
  void SetInputStd(float std_val) { input_std_ = std_val; }
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  std::string input_dtype_{"int8"};
  float input_mean_{std::numeric_limits<float>::quiet_NaN()};
  float input_std_{std::numeric_limits<float>::quiet_NaN()};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle