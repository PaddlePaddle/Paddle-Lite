#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import sys
import argparse

import math
import numpy

import paddle
import paddle.fluid as fluid


def parse_args():
    parser = argparse.ArgumentParser("fit_a_line")
    parser.add_argument(
        '--save_model',
        action='store_true',    
        help="Whether to save main program")
    parser.add_argument(
        '--num_steps',
        type=int, 
        default=1000000000000,
        help="train steps")
    parser.add_argument(
        '--num_epochs', type=int, default=100, help="number of epochs.")
    parser.add_argument(
        '--batch_size', type=int, default=20, help="batch size.")
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help="Whether to shuffle train data.")
    args = parser.parse_args()
    return args

# For training test cost
def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]


def main():
    if args.shuffle:
        print("doing shuffle")
        train_reader = paddle.batch(
                         paddle.reader.shuffle(
                             paddle.dataset.uci_housing.train(), buf_size=500),
                         batch_size=args.batch_size)
    else:
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=args.batch_size)
    
    # feature vector of length 13
    x = fluid.data(name='x', shape=[None, 13], dtype='float32')
    y = fluid.data(name='y', shape=[None, 1], dtype='float32')

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    main_program.random_seed = 90
    startup_program.random_seed = 90

    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    test_program = main_program.clone(for_test=True)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    num_epochs = args.num_epochs

    # main train loop.
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(startup_program)
    if args.save_model:
        fluid.io.save_persistables(exe, "model_dir")

        # add feed and fetch op
        feeded_var_names = ['x', 'y']
        fetch_var_names = ['mean_0.tmp_0']
        fluid.io.prepend_feed_ops(main_program, feeded_var_names)
        fluid.io.append_fetch_ops(main_program, fetch_var_names)
        with open("model_dir/__model__", "wb") as f:
            f.write(main_program.desc.serialize_to_string())

        with open("debug_main_program", "w") as f:
            f.write(str(main_program))
        print("train model saved to model_dir")
        return

    train_prompt = "Train cost"
    step = 0 
    for pass_id in range(num_epochs):
        for data_train in train_reader():
            avg_loss_value, = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_loss])
            print("%s, Step %d, Cost %f" %
                      (train_prompt, step, avg_loss_value[0]))
            if step  == args.num_steps - 1:
                return
            step += 1

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")


if __name__ == '__main__':
    args = parse_args()
    main()
