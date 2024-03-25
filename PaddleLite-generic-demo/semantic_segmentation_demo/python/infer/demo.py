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

MODEL_NAME = "pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024"
#MODEL_NAME = "pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer"
#MODEL_NAME = "pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel"
MODEL_FILE = "model.pdmodel"
PARAMS_FILE = "model.pdiparams"
CONFIG_NAME = "cityscapes_512_1024.txt"

#MODEL_NAME = "portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398"
#MODEL_NAME = "portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer"
#MODEL_NAME = "portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel"
#MODEL_FILE = "model.pdmodel"
#PARAMS_FILE = "model.pdiparams"
#CONFIG_NAME = "human_224_398.txt"

DATASET_NAME = "test"


def load_label(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            lines.append(line)
    assert len(lines) > 0, "The label file %s should not be empty!" % path
    return lines


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
    # draw_weight
    if 'draw_weight' in values:
        config['draw_weight'] = float(values['draw_weight'])
        assert config[
            'draw_weight'] >= 0, "The key 'draw_weight' should >= 0.f, but receive %f!" % config[
                'draw_weight']
        print("draw_weight: %f" % config['draw_weight'])
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


def generate_color_map(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    if num_classes < 10:
        num_classes = 10
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


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
    color_map = generate_color_map(1000)
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
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_data = rgb_image.transpose((2, 0, 1)) / 255.0
        image_data = (image_data - config['mean'].reshape(
            (3, 1, 1))) / config['std'].reshape((3, 1, 1))
        image_data = image_data.reshape(
            [1, 3, config['height'], config['width']]).astype(np.float32)
        image_tensor = fluid.core.LoDTensor()
        image_tensor.set(image_data, place)
        input_tensors['x'] = image_tensor
        # Inference
        output_tensors = exe.run(program=program,
                                 feed=input_tensors,
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
        # Postprocess
        output_data = np.array(output_tensors[0])
        # print(output_data)
        output_shape = output_data.shape
        output_rank = output_data.ndim
        if output_rank == 3:
            mask_data = output_data.astype(np.int64)
            output_height = output_shape[1]
            output_width = output_shape[2]
        elif output_rank == 4:
            mask_data = np.argmax(output_data, axis=1).astype(np.int64)
            output_height = output_shape[2]
            output_width = output_shape[3]
        else:
            print(
                "The rank of the output tensor should be 3 or 4, but receive %d!"
                % output_rank)
            exit(-1)
        mask_image = resized_image.copy()
        for j in range(output_height):
            for k in range(output_width):
                class_id = mask_data[0, j, k]
                if class_id != 0:  #  Not background
                    mask_image[j, k] = [
                        color_map[class_id][2], color_map[class_id][1],
                        color_map[class_id][0]
                    ]  # RGB->BGR
        mask_image = cv2.resize(
            mask_image, (origin_image.shape[1], origin_image.shape[0]),
            fx=0,
            fy=0,
            interpolation=cv2.INTER_CUBIC)
        origin_image = cv2.addWeighted(origin_image, 1 - config['draw_weight'],
                                       mask_image, config['draw_weight'], 0)
        cv2.imwrite(output_path, origin_image)
    print("Done.")


if __name__ == '__main__':
    main()
