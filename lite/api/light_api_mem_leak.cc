#include "lite/api/light_api.h"
#include <gperftools/heap-profiler.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include <iostream>    

int main() {
  HeapProfilerStart("Test1");
  for (size_t i = 0; i < 10000; ++i) {
    const std::string model_path{"/shixiaowei02/Paddle-Lite-v2.7/Paddle-Lite/build_opt/mobilev1_opt_out.nb"};
    {
      paddle::lite::LightPredictor predictor(model_path, false);
    }
    if (i % 100 == 0) {
      HeapProfilerDump("here");
    }
  }
  HeapProfilerStop();
  return 0;
}