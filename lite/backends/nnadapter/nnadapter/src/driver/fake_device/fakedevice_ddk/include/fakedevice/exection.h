// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <vector>
namespace fakedevice {
namespace nn {

/** Input info used by exector->SetInputs(inputs)
 *
*/
struct InputInfo {
  /// the input index, filled by user
  uint32_t index;
  /// input buffer, users should manage this buffer by themselves, including
  /// allocation and release.
  void* buf;
  /// the size of input buffer, filled by user
  uint32_t size;
  /// precision type of input data, filled by user
  int type;
  /** Layout of input data, filled by user.
   *  Currently the internal input format of FAKE_DEVICE is NCHW by default.
   *  so entering NCHW data can avoid the format conversion in the driver.
  */
  int layout;
};

/** Output info used by exector->GetOutputs(outputs)
 *
*/
struct OutputInfo {
  /// the output index, filled by user
  uint32_t index;
  /// output buffer, users should manage this buffer by themselves, including
  /// allocation and release.
  void* buf;
  /// the size of output buffer, filled by user
  uint32_t size;
  /// precision type of output data, filled by DDK
  int type;
  /// Layout of output data, filled by DDK
  int layout;
};

/** Create a exection on device
 *  You need a graph before you create a exection.
 *  The overall process is as follows:
 *  <pre>
 *  graph = new Graph();
 *  ...
 *  exector = new Exection(graph);
 *  exector->Build();
 *
 *  If no errors are encountered, the corresponding model will be created on the
 * FAKE_DEVICE.
 *  In this way, you can start inference on the FAKE_DEVICE.
 *
 *  while(something) {
 *    exector->SetInputs(inputs);
 *    exector->Run();
 *    exector->GetOutputs(outputs);
 *  }
 * </pre>
 *
 */
class Exection {
 private:
  Graph* graph_;
  /// private data
  void* device_handle;

 public:
  explicit Exection(Graph* graph);
  ~Exection();

  /** Get the graph used by Exection
   *
   *  @return the graph
   */
  Graph* GetGraph() { return graph_; }

  /** Create graph on FAKE_DEVICE.
   *
   *  @return FAKE_DEVICE_SUCCESS when success
   */
  int Build();
  int Build(fakedevice_model_buffer* fm);

  /** Set model inputs.
   *  When using a quantitative model, the input data must also be the quantized
   * data.
   *  When calling SetInputs() multiple times in succession, only the last data
   * takes effect.
   *
   *  @param inputs [in] input data
   *  @return FAKE_DEVICE_SUCCESS when success
   */
  int SetInputs(std::vector<InputInfo> inputs);

  /** Do inference on FAKE_DEVICE.
   *  This function should be called after the SetInputs() function.
   *
   *  @return FAKE_DEVICE_SUCCESS when success
   */
  int Run();

  /** Get model outputs.
   *  This function should be called after the Run() function.
   *  If using a quantitative model, the output is quantized data
   *
   *  @return FAKE_DEVICE_SUCCESS when success
   */
  int GetOutputs(std::vector<OutputInfo> outputs);
};
}  // namespace nn
}  // namespace fakedevice
