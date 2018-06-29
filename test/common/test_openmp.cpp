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

//#include <omp.h>
#include <iostream>

int main(void) {
#ifdef PADDLE_MOBILE_USE_OPENMP
  #pragma omp parallel num_threads(2)
  {
    //        int thread_id = omp_get_thread_num();
    //        int nthreads = omp_get_num_threads();
    //        std::cout << "Hello, OMP " << thread_id << "/" << nthreads <<
    //        "\n";
  }
#endif
  return 0;
}
