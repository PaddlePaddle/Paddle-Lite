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

MODEL_NAME = "ssd_mobilenet_v1_relu_voc_fp32_300"
#MODEL_NAME = "ssd_mobilenet_v1_relu_voc_int8_300_per_layer"
MODEL_FILE = ""
PARAMS_FILE = ""
CONFIG_NAME = "ssd_voc_300.txt"

#MODEL_NAME = "yolov3_mobilenet_v1_270e_coco_fp32_608"
#MODEL_FILE = "model"
#PARAMS_FILE = "params"
#CONFIG_NAME = "yolov3_coco_608.txt"

#MODEL_NAME = "picodet_relu6_int8_416_per_channel"
#MODEL_FILE = "model"
#PARAMS_FILE = "params"
#CONFIG_NAME = "picodet_coco_416.txt"

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
    # type
    assert 'type' in values, "Missing the key 'type'!"
    config['type'] = int(values['type'])
    assert config['type'] >= 1 and config[
        'type'] <= 3, "The key 'type' only supports 1,2 or 3, but receive %d!" % config[
            'type']
    print("type: %d" % config['type'])
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
    # label_list
    if 'label_list' in values:
        label_list = values['label_list']
        if len(label_list) > 0:
            config['label_list'] = load_label(dir + "/" + label_list)
            print("label_list: %d" % len(config['label_list']))
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
    color_map = generate_color_map(len(config['label_list']))
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
        if config['type'] == 2 or config['type'] == 3:
            scale_factor_data = np.array([
                config['height'] / float(origin_image.shape[0]),
                config['width'] / float(origin_image.shape[1])
            ]).reshape(1, 2).astype(np.float32)
            scale_factor_tensor = fluid.core.LoDTensor()
            scale_factor_tensor.set(scale_factor_data, place)
            input_tensors['scale_factor'] = scale_factor_tensor
        if config['type'] == 3:
            im_shape_data = np.array(
                [config['height'], config['width']]).reshape(
                    1, 2).astype(np.float32)
            im_shape_tensor = fluid.core.LoDTensor()
            im_shape_tensor.set(im_shape_data, place)
            input_tensors['im_shape'] = im_shape_tensor
        # Inference
        output_tensors = exe.run(program=program,
                                 feed=input_tensors,
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
        # Postprocess
        output_data = np.array(output_tensors[0])
        # print(output_data)
        output_index = 0
        for j in range(len(output_data)):
            class_id = int(output_data[j][0])
            score = output_data[j][1]
            if score < config['draw_threshold']:
                continue
            class_name = config['label_list'][
                class_id] if class_id >= 0 and class_id < len(config[
                    'label_list']) else 'Unknown'
            x0 = output_data[j][2]
            y0 = output_data[j][3]
            x1 = output_data[j][4]
            y1 = output_data[j][5]
            print("[%d] class_name=%s score=%f bbox=[%f,%f,%f,%f]" %
                  (output_index, class_name, score, x0, y0, x1, y1))
            if config['type'] == 1:
                x0 = x0 * origin_image.shape[1]
                y0 = y0 * origin_image.shape[0]
                x1 = x1 * origin_image.shape[1]
                y1 = y1 * origin_image.shape[0]
            lx = max(int(x0), 0)
            ly = max(int(y0), 0)
            w = max(min(int(x1), origin_image.shape[1] - 1) - lx, 0)
            h = max(min(int(y1), origin_image.shape[0] - 1) - ly, 0)
            if w > 0 and h > 0:
                color = color_map[class_id % len(color_map)]
                cv2.rectangle(origin_image, [lx, ly, w, h], color)
                cv2.rectangle(origin_image, (lx, ly), (lx + w, ly - 10), color,
                              int(-1))
                cv2.putText(origin_image, "%d.%s:%f" %
                            (output_index, class_name, score), (lx, ly),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            output_index = output_index + 1
        cv2.imwrite(output_path, origin_image)
    print("Done.")


if __name__ == '__main__':
    main()
