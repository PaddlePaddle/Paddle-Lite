// Insert fp32_to_fp16 or fp32_to_uint8 calib nodes immediately before fetch.
#include "lite/core/optimizer/mir/output_calib_pass.h"
#include <string>
#include "lite/core/op_registry.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/type_precision_cast_pass.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

TargetType PickTarget(const std::unique_ptr<SSAGraph>& graph) {
  for (const auto& place : graph->valid_places()) {
    if (place.target == TARGET(kX86)) return TARGET(kX86);
    if (place.target == TARGET(kARM)) return TARGET(kARM);
  }
  return TARGET(kHost);
}

}  // namespace

void OutputCalibPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  if (output_precision_ != PRECISION(kFP16) &&
      output_precision_ != PRECISION(kUInt8)) {
    LOG(WARNING) << "output_calib_pass: precision not set or unsupported, skip";
    return;
  }

  const char* alias = (output_precision_ == PRECISION(kFP16)) ? "fp32_to_fp16"
                                                              : "fp32_to_uint8";
  const auto target = PickTarget(graph);

  for (auto* node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt() || node->stmt()->op_type() != "fetch") continue;
    if (node->inlinks.empty()) continue;

    auto* in_arg = node->inlinks.front();
    if (!in_arg->IsArg() || in_arg->arg()->is_weight) continue;

    const std::string in_name = in_arg->AsArg().name;
    // Skip only if this fetch already ends with the *desired* output calib.
    // With enable_fp16, fetch is often fed by fp16_to_fp32 (precision_trans);
    // we still need fp32_to_uint8/fp32_to_fp16 after that float tensor.
    if (!in_arg->inlinks.empty()) {
      auto* pred = in_arg->inlinks.front();
      if (pred->IsStmt() && pred->stmt()->op_type() == "calib") {
        const auto& kernels = pred->stmt()->kernels();
        if (!kernels.empty() && kernels.front()->alias() == alias) {
          LOG(INFO) << "output_calib_pass: skip " << in_name
                    << " (desired calib " << alias << " already present)";
          continue;
        }
      }
    }

    const std::string out_name = in_name + "/out_calib";
    auto* scope = node->stmt()->op()->scope();

    auto* out_arg = graph->NewArgumentNode(out_name);
    out_arg->AsArg().type = LiteType::GetTensorTy(
        target, output_precision_, DATALAYOUT(kNCHW));
    scope->Var(out_name)->GetMutable<Tensor>()->set_precision(output_precision_);

    auto* calib_inst = graph->NewInstructNode();
    auto calib_op = LiteOpRegistry::Global().Create("calib");
    CHECK(calib_op) << "create calib op failed";
    cpp::OpDesc op_desc;
    op_desc.SetType("calib");
    op_desc.SetInput("Input", {in_name});
    op_desc.SetOutput("Out", {out_name});
    op_desc.SetAttr("scale", 1.0f);
    calib_op->Attach(op_desc, scope);
    // Opt valid_places are often float-only. ARM fp32_to_fp16 is registered at
    // PRECISION(kFP16); x86 fp32_to_fp16/uint8 at PRECISION(kFloat). Expand.
    std::vector<Place> places = graph->valid_places();
    places.emplace_back(TARGET(kARM), PRECISION(kFP16), DATALAYOUT(kNCHW));
    places.emplace_back(TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW));
    places.emplace_back(TARGET(kX86), PRECISION(kFloat), DATALAYOUT(kNCHW));
    places.emplace_back(TARGET(kX86), PRECISION(kFP16), DATALAYOUT(kNCHW));
    places.emplace_back(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    calib_op->SetValidPlaces(places);
    auto all_kernels = calib_op->CreateKernels(places);
    std::vector<std::unique_ptr<KernelBase>> picked_kernels;
    auto try_pick = [&](TargetType want) {
      for (auto& kernel : all_kernels) {
        if (kernel && kernel->alias() == alias && kernel->target() == want) {
          picked_kernels.push_back(std::move(kernel));
          return true;
        }
      }
      return false;
    };
    if (!try_pick(target) && !try_pick(TARGET(kARM)) &&
        !try_pick(TARGET(kX86)) && !try_pick(TARGET(kHost))) {
      for (auto& kernel : all_kernels) {
        if (kernel && kernel->alias() == alias) {
          picked_kernels.push_back(std::move(kernel));
          break;
        }
      }
    }
    CHECK(!picked_kernels.empty())
        << "output_calib_pass: no calib kernel alias=" << alias;
    calib_inst->AsStmt("calib", std::move(picked_kernels), calib_op);

    RemoveDirectedLink(in_arg, node);
    DirectedLink(in_arg, calib_inst);
    DirectedLink(calib_inst, out_arg);
    DirectedLink(out_arg, node);

    UpdateInputs(node->stmt()->op().get(), in_name, out_name);
    OpInfo update_op_info = *node->stmt()->op_info();
    update_op_info.SetAttr<int>("data_type",
                                static_cast<int>(output_precision_));
    node->stmt()->ResetOp(update_op_info, graph->valid_places());

    LOG(INFO) << "output_calib_pass: " << in_name << " -> calib(" << alias
              << ") -> fetch data_type="
              << (output_precision_ == PRECISION(kFP16) ? "fp16" : "uint8");
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(output_calib_pass, paddle::lite::mir::OutputCalibPass)
    .BindTargets({TARGET(kAny)});