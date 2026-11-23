// Analytic integer feed + in-graph calib dequant.
// int8 (default): feed kInt8 (xor-128) + int8_to_fp16|fp32
// uint8:          feed kUInt8 (raw greyscale) + uint8_to_fp16|fp32
// Math: uint8 path (u-128)*scale ≡ int8 path after q=(int8_t)(u^128).
// scale = 1/std and the model input x = (u - mean)/std are explicit
// (--analytic_input_mean/--analytic_input_std). The calib kernels centre at
// 128, so x = (u-128)/std + (128-mean)/std; the constant is folded into the
// bias of the conv(s) that read the feed (exact except at zero-padded taps).
// Prefer *to_fp16 when kFP16 places present; else *to_fp32.
#include "lite/core/optimizer/mir/analytic_input_quant_pass.h"
#include <cmath>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
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

mir::Node* FindFirstConv2dFrom(mir::Node* start) {
  if (!start) return nullptr;
  std::queue<mir::Node*> q;
  std::set<mir::Node*> visited;
  q.push(start);
  while (!q.empty()) {
    auto* cur = q.front();
    q.pop();
    if (!cur || visited.count(cur)) continue;
    visited.insert(cur);
    for (auto* out : cur->outlinks) {
      if (out->IsStmt() && (out->stmt()->op_type() == "conv2d" ||
                            out->stmt()->op_type() == "depthwise_conv2d")) {
        return out;
      }
      // Walk through intermediate ops (e.g. fp32_to_fp16 calib from enable_fp16)
      // and non-weight args so we still find the first real conv.
      if (out->IsStmt()) {
        q.push(out);
      } else if (out->IsArg() && !out->arg()->is_weight) {
        q.push(out);
      }
    }
  }
  return nullptr;
}

// bias[o] += offset * sum(W[o, ...]): a constant added to every input pixel
// of this conv, moved to its (BN-fused) bias. Weights must still be float.
void FoldInputOffsetIntoBias(mir::Node* conv, float offset) {
  auto* op_info = conv->stmt()->op_info();
  const auto& type = conv->stmt()->op_type();
  CHECK(type == "conv2d" || type == "depthwise_conv2d")
      << "analytic_input_quant_pass: feed consumer " << type
      << " cannot absorb input mean != 128; use mean 128 or fold it in "
         "convert_mono.py";
  CHECK(op_info->HasInput("Bias") && !op_info->Input("Bias").empty())
      << "analytic_input_quant_pass: first conv has no Bias to absorb "
         "(128-mean)/std; run conv_bn fuse or use mean 128";
  auto* scope = conv->stmt()->op()->scope();
  auto* w = scope->FindVar(op_info->Input("Filter").front())
                ->GetMutable<lite::Tensor>();
  auto* b = scope->FindVar(op_info->Input("Bias").front())
                ->GetMutable<lite::Tensor>();
  auto is_quant = [](PrecisionType p) {
    return p == PRECISION(kInt8) || p == PRECISION(kInt16) ||
           p == PRECISION(kFP16);
  };
  CHECK(!is_quant(w->precision()) && !is_quant(b->precision()))
      << "analytic_input_quant_pass: mean != 128 needs float first-conv "
         "weights/bias (not with --quant_model); use mean 128";
  const int64_t out_c = w->dims()[0];
  const int64_t per = w->numel() / out_c;
  CHECK_EQ(b->numel(), out_c) << "unexpected first-conv bias size";
  const float* wd = w->data<float>();
  float* bd = b->mutable_data<float>();
  for (int64_t o = 0; o < out_c; ++o) {
    double s = 0.0;
    for (int64_t i = 0; i < per; ++i) s += wd[o * per + i];
    bd[o] += static_cast<float>(offset * s);
  }
  LOG(INFO) << "analytic_input_quant_pass: folded input offset " << offset
            << " into " << op_info->Input("Bias").front();
}

}  // namespace

void AnalyticInputQuantPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  mir::Node* feed_out = nullptr;
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt() || node->stmt()->op_type() != "feed") continue;
    for (auto* out : node->outlinks) {
      if (out->IsArg() && !out->arg()->is_weight) {
        feed_out = out;
        break;
      }
    }
    break;
  }
  if (!feed_out) {
    LOG(WARNING) << "analytic_input_quant_pass: no feed output found, skip";
    return;
  }

  // Already have our dequant after feed? (Do not skip on fp32_to_fp16 from
  // enable_fp16/type_precision_cast — that is a different cast.)
  for (auto* cons : feed_out->outlinks) {
    if (!cons->IsStmt() || cons->stmt()->op_type() != "calib") continue;
    const auto& kernels = cons->stmt()->kernels();
    if (kernels.empty()) continue;
    const auto& a = kernels.front()->alias();
    if (a == "int8_to_fp32" || a == "int8_to_fp16" || a == "uint8_to_fp32" ||
        a == "uint8_to_fp16") {
      LOG(INFO) << "analytic_input_quant_pass: " << a
                << " already after feed, skip";
      return;
    }
  }

  mir::Node* conv_node = FindFirstConv2dFrom(feed_out);
  if (!conv_node) {
    LOG(WARNING) << "analytic_input_quant_pass: no conv2d after feed, skip";
    return;
  }

  const bool use_uint8 =
      (input_dtype_ == "uint8" || input_dtype_ == "u8" ||
       input_dtype_ == "kUInt8");
  if (!use_uint8 && input_dtype_ != "int8" && input_dtype_ != "i8" &&
      input_dtype_ != "kInt8" && !input_dtype_.empty()) {
    LOG(WARNING) << "analytic_input_quant_pass: unknown input_dtype="
                 << input_dtype_ << " (use int8|uint8); defaulting to int8";
  }

  CHECK(std::isfinite(input_mean_) && std::isfinite(input_std_) &&
        input_std_ > 0.f)
      << "analytic_input_quant_pass: set --analytic_input_mean and "
         "--analytic_input_std (pixel units, x = (u - mean) / std). "
         "rec_v3/rec_numeric: 127.5/127.5; det: values from convert_mono.py "
         "mono_norm.json (128/58.395)";
  const float input_scale = 1.0f / input_std_;
  const float input_offset = (128.0f - input_mean_) / input_std_;
  const auto target = PickTarget(graph);

  const std::string in_name = feed_out->AsArg().name;
  const std::string out_name = in_name + "/analytic_in_calib";
  auto* scope = conv_node->stmt()->op()->scope();

  // Host-visible feed precision.
  const PrecisionType feed_prec =
      use_uint8 ? PRECISION(kUInt8) : PRECISION(kInt8);
  feed_out->AsArg().type =
      LiteType::GetTensorTy(TARGET(kHost), feed_prec, DATALAYOUT(kNCHW));

  // Prefer direct *→fp16 when enable_fp16 places are present (one calib,
  // one scale). Fall back to *→fp32 for pure float backbones.
  // places_have_fp16 stays true even if we fall back so we still force
  // first-conv kFP16 (avoids ResetOp demotion → dead heatmap).
  bool places_have_fp16 = false;
  for (const auto& p : graph->valid_places()) {
    if (p.precision == PRECISION(kFP16)) {
      places_have_fp16 = true;
      break;
    }
  }
  bool want_fp16 = places_have_fp16;  // may clear if *to_fp16 missing
  const char* alias =
      use_uint8 ? (want_fp16 ? "uint8_to_fp16" : "uint8_to_fp32")
                : (want_fp16 ? "int8_to_fp16" : "int8_to_fp32");
  const PrecisionType out_prec =
      want_fp16 ? PRECISION(kFP16) : PRECISION(kFloat);

  auto* out_arg = graph->NewArgumentNode(out_name);
  out_arg->AsArg().type =
      LiteType::GetTensorTy(target, out_prec, DATALAYOUT(kNCHW));
  scope->Var(out_name)->GetMutable<Tensor>()->set_precision(out_prec);

  // calib: dequant to fp16 or fp32
  auto* calib_inst = graph->NewInstructNode();
  auto calib_op = LiteOpRegistry::Global().Create("calib");
  CHECK(calib_op) << "create calib op failed";
  cpp::OpDesc op_desc;
  op_desc.SetType("calib");
  op_desc.SetInput("Input", {in_name});
  op_desc.SetOutput("Out", {out_name});
  op_desc.SetAttr("scale", input_scale);
  calib_op->Attach(op_desc, scope);
  std::vector<Place> places = graph->valid_places();
  places.emplace_back(TARGET(kARM), PRECISION(kInt8), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kX86), PRECISION(kInt8), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kHost), PRECISION(kInt8), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kARM), PRECISION(kUInt8), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kX86), PRECISION(kUInt8), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kHost), PRECISION(kUInt8), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kARM), PRECISION(kFP16), DATALAYOUT(kNCHW));
  places.emplace_back(TARGET(kX86), PRECISION(kFloat), DATALAYOUT(kNCHW));
  calib_op->SetValidPlaces(places);
  auto pick_alias = [&](const char* want_alias) {
    auto kernels = calib_op->CreateKernels(places);
    std::vector<std::unique_ptr<KernelBase>> picked;
    auto try_one = [&](TargetType want) {
      for (auto& kernel : kernels) {
        if (kernel && kernel->alias() == want_alias &&
            kernel->target() == want) {
          picked.push_back(std::move(kernel));
          return true;
        }
      }
      return false;
    };
    if (!try_one(target) && !try_one(TARGET(kARM)) &&
        !try_one(TARGET(kX86)) && !try_one(TARGET(kHost))) {
      for (auto& kernel : kernels) {
        if (kernel && kernel->alias() == want_alias) {
          picked.push_back(std::move(kernel));
          break;
        }
      }
    }
    return picked;
  };

  std::vector<std::unique_ptr<KernelBase>> picked_kernels = pick_alias(alias);
  // Graceful fallback if *to_fp16 not in this opt binary yet.
  if (picked_kernels.empty() && want_fp16) {
    const char* fallback =
        use_uint8 ? "uint8_to_fp32" : "int8_to_fp32";
    LOG(WARNING) << "analytic_input_quant_pass: no " << alias
                 << "; falling back to " << fallback << " (+ later fp16 cast)";
    alias = fallback;
    out_arg->AsArg().type =
        LiteType::GetTensorTy(target, PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->Var(out_name)->GetMutable<Tensor>()->set_precision(
        PRECISION(kFloat));
    picked_kernels = pick_alias(alias);
    want_fp16 = false;  // out is float; type_precision_cast may still fp16-cast
  }
  CHECK(!picked_kernels.empty())
      << "analytic_input_quant_pass: no calib kernel alias=" << alias;
  calib_inst->AsStmt("calib", std::move(picked_kernels), calib_op);

  // Rewire every non-weight consumer of feed_out through calib.
  std::vector<mir::Node*> consumers;
  for (auto* cons : feed_out->outlinks) {
    if (cons->IsStmt()) consumers.push_back(cons);
  }
  if (input_offset != 0.f) {
    for (auto* cons : consumers) FoldInputOffsetIntoBias(cons, input_offset);
  }
  for (auto* cons : consumers) {
    RemoveDirectedLink(feed_out, cons);
    UpdateInputs(cons->stmt()->op().get(), in_name, out_name);
    OpInfo info = *cons->stmt()->op_info();
    if (info.HasAttr("enable_int8")) {
      info.SetAttr("enable_int8", false);
    }
    info.SetAttr("analytic_input_quant", true);
    // ResetOp re-runs CreateKernels and leaves an *unsorted* list; picked
    // kernel is kernels.front(). That often demotes an already-selected
    // kFP16 first-conv to kFloat. Then type_precision_cast inserts
    // fp16_to_fp32 (or int8→fp16→fp32) before a float first-conv while the
    // rest of the graph stays kFP16 — dead heatmap (constant floor).
    // When we produce fp16 activations, force consumers to keep kFP16.
    cons->stmt()->ResetOp(info, graph->valid_places());
    if (places_have_fp16) {
      auto& kernels = cons->stmt()->kernels();
      size_t fp16_idx = kernels.size();
      for (size_t i = 0; i < kernels.size(); ++i) {
        if (kernels[i] && kernels[i]->precision() == PRECISION(kFP16)) {
          fp16_idx = i;
          break;
        }
      }
      if (fp16_idx < kernels.size() && fp16_idx != 0) {
        std::swap(kernels[0], kernels[fp16_idx]);
      }
      if (!kernels.empty() && kernels.front() &&
          kernels.front()->precision() == PRECISION(kFP16)) {
        // Drop non-fp16 candidates so later passes cannot re-prefer float.
        std::vector<std::unique_ptr<KernelBase>> only_fp16;
        only_fp16.push_back(std::move(kernels.front()));
        kernels = std::move(only_fp16);
      } else {
        LOG(WARNING) << "analytic_input_quant_pass: consumer "
                     << cons->stmt()->op_type()
                     << " has no kFP16 kernel after ResetOp; place="
                     << (kernels.empty() ? "none"
                                         : kernels.front()->summary());
      }
    }
    DirectedLink(out_arg, cons);
  }

  DirectedLink(feed_out, calib_inst);
  DirectedLink(calib_inst, out_arg);

  LOG(INFO) << "analytic_input_quant_pass: feed("
            << (use_uint8 ? "kUInt8" : "kInt8") << ")->calib(" << alias
            << ",scale=" << input_scale << ")->" << (want_fp16 ? "fp16" : "fp32")
            << " ops; mean=" << input_mean_ << " std=" << input_std_;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(analytic_input_quant_pass,
                  paddle::lite::mir::AnalyticInputQuantPass)
    .BindTargets({TARGET(kAny)});
