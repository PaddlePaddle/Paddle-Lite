// Copyright (c) 2019 Cambricon Authors. All Rights Reserved.

#include <Eigen/Core>
#include "lite/backends/mlu/target_wrapper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

using TargetW = TargetWrapper<TARGET(kMLU)>;

// Host to MLU memory.
void CopyFromHostSync(void* target, const void* source, size_t size) {
  TargetW::MemcpySync(target, source, size, IoDirection::HtoD);
}

// MLU to Host memory.
void CopyToHostSync(void* target, const void* source, size_t size) {
  TargetW::MemcpySync(target, source, size, IoDirection::DtoH);
}

/*
 * This kernel copies a tensor from host to MLU space.
 */
template <PrecisionType Precision>
class IoCopyHostToMluCompute
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  using handler_t = KernelBase::type_infer_handler_t;
  using param_t = operators::IoCopyParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kX86));
    auto mem_size = param.x->memory_size();
    // LOG(INFO) << "copy size " << mem_size;
    auto* data = param.y->mutable_data(TARGET(kMLU), mem_size);
    param.y->set_precision(param.x->precision());
    CopyFromHostSync(data, param.x->raw_data(), mem_size);
  }

  std::unique_ptr<handler_t> GetTypeInferHandler() override {
    std::unique_ptr<handler_t> res(new handler_t);
    *res = [](const std::map<std::string, const Type*>& inputs,
              const std::string& out) -> const Type* {
      CHECK(!inputs.empty());
      auto* type = inputs.at("Input");
      CHECK(type->target() == TARGET(kHost));

      auto out_place = type->place();
      out_place.target = TARGET(kMLU);
      auto* out_type = Type::Get(type->id(),
                                 out_place.target,
                                 out_place.precision,
                                 out_place.layout,
                                 out_place.device);
      return out_type;
    };
    return res;
  }

  std::string doc() const override { return "Copy IO from HOST to MLU"; }
};

/*
 * This kernel copies a tensor from MLU to host space.
 */
template <PrecisionType Precision>
class IoCopyMluToHostCompute
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  void Run() override {
    auto& param = this->template Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kMLU));
    auto mem_size = param.x->memory_size();
    auto* data = param.y->mutable_data(TARGET(kHost), mem_size);

    // sync queue to ensure process done
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    CNRT_CALL(cnrtSyncQueue(mlu_context.exec_queue()));

    CopyToHostSync(data, param.x->raw_data(), mem_size);
  }

  std::string doc() const override { return "Copy IO from MLU to HOST"; }
};

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    io_copy,
    kMLU,
    kFloat,
    kNHWC,
    paddle::lite::kernels::mlu::IoCopyHostToMluCompute<PRECISION(kFloat)>,
    host_to_device_kFloat)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMLU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    io_copy,
    kMLU,
    kFP16,
    kNHWC,
    paddle::lite::kernels::mlu::IoCopyHostToMluCompute<PRECISION(kFP16)>,
    host_to_device_kFP16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMLU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    io_copy,
    kMLU,
    kFloat,
    kNHWC,
    paddle::lite::kernels::mlu::IoCopyMluToHostCompute<PRECISION(kFloat)>,
    device_to_host_kFloat)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kMLU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(
    io_copy,
    kMLU,
    kFP16,
    kNHWC,
    paddle::lite::kernels::mlu::IoCopyMluToHostCompute<PRECISION(kFP16)>,
    device_to_host_kFP16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kMLU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
