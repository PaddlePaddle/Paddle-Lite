// Insert calib before fetch so output tensors are kFP16 or kUInt8.
#pragma once
#include <memory>
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

class OutputCalibPass : public ProgramPass {
 public:
  void SetOutputPrecision(PrecisionType p) { output_precision_ = p; }
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  PrecisionType output_precision_{PRECISION(kUnk)};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle