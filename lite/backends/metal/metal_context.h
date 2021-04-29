// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_BACKENDS_METAL_METAL_CONTEXT_H_
#define LITE_BACKENDS_METAL_METAL_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
class RuntimeProgram;

class MetalContext {
 public:
	MetalContext();
	~ MetalContext();
  /// device
  void PrepareDevices();
  int GetDevicesNum();
  void* GetDeviceByID(int id);
  const void* GetDefaultDevice();

  void CreateCommandBuffer(RuntimeProgram* program = nullptr);
  void WaitUntilCompleted();

  void set_metal_path(std::string path);
	void set_use_aggressive_optimization(bool flag){};
	void set_use_mps(bool flag){};
  bool use_mps() const { return true; }
	bool use_quadruple() const { return false; }
  bool use_winograde() const { return false; }

	void *backend() const {
			return mContext;
	}
	
	RuntimeProgram *program() const {
			return program_;
	}
	
 private:
  void* mContext = nullptr;
  bool got_devices_{false};
  std::string metal_path_;
  RuntimeProgram* program_ = nullptr;
};
}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_CONTEXT_H_
