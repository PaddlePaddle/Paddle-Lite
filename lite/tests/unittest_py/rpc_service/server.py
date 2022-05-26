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
import os
import re
import shutil
from rpyc.utils.server import ThreadedServer
import paddlelite
from paddlelite.lite import *
import copy
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
import argparse
import platform
parser = argparse.ArgumentParser()
parser.add_argument(
    "--server_ip",
    default="localhost",
    type=str,
    help="when rpc is used , the ip address of the server")

args = parser.parse_args()


def ParsePlaceInfo(place_str):
    # todo: this func should be completed later
    infos = ''.join(place_str.split()).split(",")
    if len(infos) == 1:
        if infos[0] in TargetType.__members__:
            return Place(eval("TargetType." + infos[0]))
        else:
            logging.error("Error place info: " + place_str)
    elif len(infos) == 2:
        if (infos[0] in TargetType.__members__) and (
                infos[1] in PrecisionType.__members__):
            return Place(
                eval("TargetType." + infos[0]),
                eval("PrecisionType." + infos[1]))
        else:
            logging.error("Error place info: " + place_str)
    elif len(infos) == 3:
        if (infos[0] in TargetType.__members__) and (
                infos[1] in PrecisionType.__members__) and (
                    infos[2] in DataLayoutType.__members__):
            return Place(
                eval("TargetType." + infos[0]),
                eval("PrecisionType." + infos[1]),
                eval("DataLayoutType." + infos[2]))
        else:
            logging.error("Error place info: " + place_str)
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
        lite_config.set_threads(config["thread"])
    if "discarded_passes" in config:

        for discarded_pass in config["discarded_passes"]:
            lite_config.add_discarded_pass(discarded_pass)
    return lite_config


class RPCService(rpyc.Service):
    def exposed_run_lite_model(self, model, params, inputs, config_str):
        '''
        Test a single case.
        '''
        # 1. store original model
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        self.cache_dir = os.path.join(abs_dir,
                                      str(self.__module__) + '_cache_dir')
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        with open(self.cache_dir + "/model", "wb") as f:
            f.write(model)
        with open(self.cache_dir + "/params", "wb") as f:
            f.write(params)
        # 2. run inference
        config = ParsePaddleLiteConfig(self, config_str)
        config.set_model_buffer(model, len(model), params, len(params))
        #  2.1 metal configs
        module_path = os.path.dirname(paddlelite.__file__)
        config.set_metal_lib_path(module_path + "/libs/lite.metallib")
        config.set_metal_use_mps(True)

        predictor = create_paddle_predictor(config)

        # 3. optimized model
        predictor.save_optimized_pb_model(self.cache_dir + "/opt_model")
        with open(self.cache_dir + "/opt_model/model", "rb") as f:
            model = f.read()

        for name in inputs:
            input_tensor = predictor.get_input_by_name(name)
            input_tensor.from_numpy(inputs[name]['data'])
            if inputs[name]['lod'] is not None:
                input_tensor.set_lod(inputs[name]['lod'])

        predictor.run()
        # 4. inference results
        result = {}
        for out_name in predictor.get_output_names():
            result[out_name] = predictor.get_output_by_name(out_name).numpy()
        result_res = copy.deepcopy(result)
        return result_res, model


if __name__ == "__main__":
    paddle_lite_path = os.path.abspath(__file__)
    paddlelite_source_path = re.findall(r"(.+?)Paddle-Lite",
                                        paddle_lite_path)[0]
    rpc_port_file = paddlelite_source_path + "Paddle-Lite/lite/tests/unittest_py/rpc_service/.port_id"
    port_id = int(open(rpc_port_file).read())
    server = ThreadedServer(RPCService, port=port_id, hostname=args.server_ip)
    server.start()
