# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rpyc
from rpyc.utils.server import ThreadedServer
from paddlelite.lite import *

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


# TargetType
target_type_map = {
      "Host": TargetType.Host,
      "X86": TargetType.X86,
      "CUDA": TargetType.CUDA,
      "ARM": TargetType.ARM,
      "OpenCL": TargetType.OpenCL,
      "FPGA": TargetType.FPGA,
      "NPU": TargetType.NPU,
      "MLU": TargetType.MLU,
      "RKNPU": TargetType.RKNPU,
      "APU": TargetType.APU,
      "HUAWEI_ASCEND_NPU": TargetType.HUAWEI_ASCEND_NPU,
      "IMAGINATION_NNA": TargetType.IMAGINATION_NNA,
      "INTEL_FPGA": TargetType.INTEL_FPGA,
      "Any": TargetType.Any}

# PrecisionType
precision_type_map = {
      "FP16": PrecisionType.FP16,
      "FP32": PrecisionType.FP32,
      "FP64": PrecisionType.FP64,
      "UINT8": PrecisionType.UINT8,
      "INT8": PrecisionType.INT8,
      "INT16": PrecisionType.INT16,
      "INT32": PrecisionType.INT32,
      "INT64": PrecisionType.INT64,
      "BOOL": PrecisionType.BOOL,
       "Any": PrecisionType.Any}


# DataLayoutType
data_layout_map = {
      "NCHW": DataLayoutType.NCHW,
      "NHWC": DataLayoutType.NHWC,
      "ImageDefault": DataLayoutType.ImageDefault,
      "ImageFolder": DataLayoutType.ImageFolder,
      "ImageNW": DataLayoutType.ImageNW,
      "Any": DataLayoutType.Any}



def ParsePlaceInfo(place_str):
   # todo: this func should be completed later
   infos = ''.join(place_str.split()).split(",")
   if len(infos) == 1 :
       if infos[0] in target_type_map:
           return Place(target_type_map[infos[0]])
       else:
           logging.error("Error place info: " + place_str)
   elif len(infos) == 2 :
       if (infos[0] in target_type_map) and (infos[1] in precision_type_map):
           return Place(target_type_map[infos[0]], precision_type_map[infos[1]])
       else:
           logging.error("Error place info: " + place_str)
   elif len(infos) == 3 :
       if (infos[0] in target_type_map) and (infos[1] in precision_type_map) and (infos[2] in data_layout_map):
           return Place(target_type_map[infos[0]], precision_type_map[infos[1]], data_layout_map[infos[2]])
       else:
           logging.error("Error place info: " + place_str)

def ParsePaddleLiteConfig(self, config):
    lite_config = CxxConfig()
    if "valid_targets" in config:
        valid_places = []
        for place_str in config["valid_targets"]:
            valid_places.append(ParsePlaceInfo(place_str))
        lite_config.set_valid_places(valid_places)
    if "thread" in config:
        lite_config.set_thread(pred_config["thread"])
    return lite_config




class RPCService(rpyc.Service):
    def exposed_run_lite_model(self, model, params,inputs):
        '''
        Test a single case.
        '''
        config = ParsePaddleLiteConfig(self, pred_config)
        config.set_model_buffer(model, len(model), params, len(params))
        predictor = create_paddle_predictor(config)
        for name in inputs:
            input_tensor = predictor.get_input_by_name(name)
            input_tensor.from_numpy(inputs[name]['data'])
            if inputs[name]['lod'] is not None:
                input_tensor.set_lod(inputs[name]['lod'])
        predictor.run()
        result = {}
        for out_name in predictor.get_output_names():
            result[out_name] = predictor.get_output_by_name(out_name).numpy()
        return result


if __name__ == "__main__":
    server = ThreadedServer(RPCService, port =18812, protocol_config = rpyc.core.protocol.DEFAULT_CONFIG)
    server.start()
