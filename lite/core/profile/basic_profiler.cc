// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/core/profile/basic_profiler.h"

DEFINE_string(time_profile_file,
              "time_profile.txt",
              "Lite time profile information dump file");

DEFINE_string(time_profile_summary_file,
              "time_profile_summary.txt",
              "Lite time profile summary information dump file");

namespace paddle {
namespace lite {
namespace profile {

const int BasicTimer::data_w = 10;
const int BasicTimer::name_w = 15;

template class BasicProfiler<BasicTimer>;

template <typename TimerT>
BasicProfiler<TimerT>::~BasicProfiler() {
  LOG(INFO) << "Basic Profile dumps:";
  auto b_repr = TimerT::basic_repr_header() + "\n" + basic_repr();
  LOG(INFO) << "\n" + b_repr;

  // Dump to file
  std::ofstream basic_ostream(FLAGS_time_profile_file);
  CHECK(basic_ostream.is_open()) << "Open " << FLAGS_time_profile_file
                                 << " failed";
  basic_ostream.write(b_repr.c_str(), b_repr.size());
  basic_ostream.close();

  LOG(INFO) << "Summary Profile dumps:";
  auto s_repr = summary_repr_header() + "\n" + summary_repr();
  LOG(INFO) << "\n" + s_repr;

  // Dump to file
  std::ofstream summary_ostream(FLAGS_time_profile_summary_file);
  CHECK(summary_ostream.is_open()) << "Open " << FLAGS_time_profile_summary_file
                                   << " failed";
  summary_ostream.write(s_repr.c_str(), s_repr.size());
  summary_ostream.close();
}

}  // namespace profile
}  // namespace lite
}  // namespace paddle
