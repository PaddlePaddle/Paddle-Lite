# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import cv2
import shutil
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()

MODEL_NAME = "tinypose_fp32_128_96"
#MODEL_NAME = "tinypose_int8_128_96_per_channel"
MODEL_FILE = "model.pdmodel"
PARAMS_FILE = "model.pdiparams"
CONFIG_NAME = "tinypose_128_96.txt"

#MODEL_NAME = "tinypose_fp32_256_192"
#MODEL_FILE = "model.pdmodel"
#PARAMS_FILE = "model.pdiparams"
#CONFIG_NAME = "tinypose_256_192.txt"

DATASET_NAME = "test"


def load_config(path):
    values = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            value = line.split(':')
            assert len(
                value
            ) == 2, "Format error at '%s', it should be '<key>:<value>'." % line
            values[value[0]] = value[1]
    dir = os.path.dirname(path)
    print("dir: %s" % dir)
    config = {}
    # width
    assert 'width' in values, "Missing the key 'width'!"
    config['width'] = int(values['width'])
    assert config[
        'width'] > 0, "The key 'width' should > 0, but receive %d!" % config[
            'width']
    print("width: %d" % config['width'])
    # height
    assert 'height' in values, "Missing the key 'height'!"
    config['height'] = int(values['height'])
    assert config[
        'height'] > 0, "The key 'height' should > 0, but receive %d!" % config[
            'height']
    print("height: %d" % config['height'])
    # mean
    assert 'mean' in values, "Missing the key 'mean'!"
    config['mean'] = np.array(values['mean'].split(','), dtype=np.float32)
    assert config[
        'mean'].size == 3, "The key 'mean' should contain 3 values, but receive %ld!" % config[
            'mean'].size
    print("mean: %f,%f,%f" %
          (config['mean'][0], config['mean'][1], config['mean'][2]))
    # std
    assert 'std' in values, "Missing the key 'std'!"
    config['std'] = np.array(values['std'].split(','), dtype=np.float32)
    assert config[
        'std'].size == 3, "The key 'std' should contain 3 values, but receive %ld!" % config[
            'std'].size
    print("std: %f,%f,%f" %
          (config['std'][0], config['std'][1], config['std'][2]))
    # draw_threshold
    if 'draw_threshold' in values:
        config['draw_threshold'] = float(values['draw_threshold'])
        assert config[
            'draw_threshold'] >= 0, "The key 'draw_threshold' should >= 0.f, but receive %f!" % config[
                'draw_threshold']
        print("draw_threshold: %f" % config['draw_threshold'])
    # use_dark_decode
    if 'use_dark_decode' in values:
        config['use_dark_decode'] = False if values['use_dark_decode'].lower(
        ) == 'false' or values['use_dark_decode'].lower() == '0' else True
        print("use_dark_decode: %r" % config['use_dark_decode'])
    return config


def load_dataset(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            lines.append(line)
    assert len(
        lines) > 0, "The dataset list file %s should not be empty!" % path
    return lines


def main(argv=None):
    # Load model
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    model_dir = "../../assets/models/" + MODEL_NAME
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_dir, exe)
    else:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             model_dir,
             exe,
             model_filename=MODEL_FILE,
             params_filename=PARAMS_FILE)
    print("--- feed_target_names ---")
    print(feed_target_names)
    print("--- fetch_targets ---")
    print(fetch_targets)
    # Parse the config file to extract the model info
    config = load_config("../../assets/configs/" + CONFIG_NAME)
    # print(config)
    # Load dataset list
    dataset = load_dataset("../../assets/datasets/" + DATASET_NAME +
                           "/list.txt")
    # Traverse the list of the dataset and run inference on each sample
    sample_count = len(dataset)
    shutil.rmtree("../../assets/datasets/" + DATASET_NAME + "/outputs")
    os.mkdir("../../assets/datasets/" + DATASET_NAME + "/outputs")
    for i in range(sample_count):
        sample_name = dataset[i]
        print("[%u/%u] Processing %s" % (i + 1, sample_count, sample_name))
        input_path = "../../assets/datasets/" + DATASET_NAME + "/inputs/" + sample_name
        output_path = "../../assets/datasets/" + DATASET_NAME + "/outputs/" + sample_name
        # Preprocess
        input_tensors = {}
        origin_image = cv2.imread(input_path)
        resized_image = cv2.resize(
            origin_image, (config['width'], config['height']),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_CUBIC)
        image_data = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_data = image_data.transpose((2, 0, 1)) / 255.0
        image_data = (image_data - config['mean'].reshape(
            (3, 1, 1))) / config['std'].reshape((3, 1, 1))
        image_data = image_data.reshape(
            [1, 3, config['height'], config['width']]).astype(np.float32)
        image_tensor = fluid.core.LoDTensor()
        image_tensor.set(image_data, place)
        input_tensors['image'] = image_tensor
        # Inference
        output_tensors = exe.run(program=program,
                                 feed=input_tensors,
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
        # Postprocess
        heatmap_data = np.array(output_tensors[0])
        print(heatmap_data)
        index_data = np.array(output_tensors[1])
        print(index_data)
        # TODO(hong19860320) Add postprocess
        cv2.imwrite(output_path, origin_image)
    print("Done.")


if __name__ == '__main__':
    main()
