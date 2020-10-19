# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import re
import argparse
import subprocess

def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ops_path', default='ops.txt', help='Input ops path.')
    parser.add_argument(
        '--latency_lookup_table_path',
        default='latency_lookup_table.txt',
        help='Output ops latency path.')
    parser.add_argument(
        '--platform', default='android', help='Platform: android/ios/custom.')
    parser.add_argument('--threads', type=int, default=1, help='Threads.')
    parser.add_argument('--power_mode', type=int, default=0, help='PowerMode.')
    parser.add_argument('--warmup_times', type=int, default=5, 
        help='Warm up times of op when estimating latency.')
    parser.add_argument('--repeats_times', type=int, default=100,
        help='Running times of op when estimating latency.')
    parser.add_argument('--arm_v7_v8', type=str, default='armv8',
        help='Indicate arm architecture v7 or v8.')
    args = parser.parse_args()
    return args

def check_dev_connect():
    cmd = 'adb devices | grep device'
    dev_info = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = dev_info.communicate()[0]
    res = out.decode().find("\tdevice")
    if res == -1:
        print("No android device is attached")
        sys.exit()

def get_dev_info():
    cmd = 'adb shell "cat /proc/cpuinfo | grep Hardware"'
    dev_info = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = dev_info.communicate()[0]
    out = out.decode().strip('\n')
    dev_info = out.strip('Hardware\t:').strip()
    cmd = 'adb shell "cat /proc/cpuinfo | grep part"'
    cpu_info = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = cpu_info.communicate()[0]
    out = (out.decode().strip('\n').split('\n'))
    core_num = len(out)
    arch_type = ['UNKNOWN CPU ARCH']*core_num
    for i, v in enumerate(out):
        out = v.strip('CPU part').strip().strip(':').strip()
        if out == '0xd03':
            arch_type[i] = 'ARM_A53'
        elif out == '0xd05':
            arch_type[i] = 'ARM_A55'
        elif out == '0xd07':
            arch_type[i] = 'ARM_A57'
        elif out == '0xd08':
            arch_type[i] = 'ARM_A72'
        elif out == '0xd09':
            arch_type[i] = 'ARM_A73'
        elif out == '0xd0a':
            arch_type[i] = 'ARM_A75'
        elif out == '0xd40':
            arch_type[i] = 'ARM_A76'
        elif out == '0x804':
            # 855
            arch_type[i] = 'ARM_A76'
        elif out == '0x805':
            # 855
            arch_type[i] = 'ARM_A55'
        elif out == '0x802':
            # 845
            arch_type[i] = 'ARM_A75'
        elif out == '0x803':
            # 845
            arch_type[i] = 'ARM_A55'
        elif out == '0x801':
            # 835
            arch_type[i] = 'ARM_A73'
        elif out == '0x800':
            # 835
            arch_type[i] = 'ARM_A73'
        elif out == '0x205':
            # 820
            arch_type[i] = 'ARM_A72'
        else:
            arch_type[i] = 'UNKNOWN CPU ARCH'
    return dev_info, core_num, arch_type

def get_op_latency(op, platform):
    """Get model latency.

    Args:
        op: list, a list of str represents the op and its parameters.
        platform: str, platform name.

    Returns:
        float, op latency.
    """
    if platform == 'android':
        commands = 'adb shell "cd /data/local/tmp/bin && ./get_{}_latency {}"'.format(
            op[0], ' '.join(op[1:]))
        proc = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
        out = proc.communicate()[0]
        avg_out = [_ for _ in out.decode().split('\n') if 'Avg Latency' in _][-1]
        avg_out = re.findall(r'\d+\.?\d*', avg_out)[0]
        avg_out = float(avg_out)
        min_out = [_ for _ in out.decode().split('\n') if 'Min Latency' in _][-1]
        min_out = re.findall(r'\d+\.?\d*', min_out)[0]
        min_out = float(min_out)
        max_out = [_ for _ in out.decode().split('\n') if 'Max Latency' in _][-1]
        max_out = re.findall(r'\d+\.?\d*', max_out)[0]
        max_out = float(max_out)
    elif platform == 'ios':
        print('ios platform is not supported now')
        sys.exit()
    else:
        print('Please define `get_op_latency` for {} platform'.format(platform))
        sys.exit()
    return avg_out, min_out, max_out

def main():
    args = get_args()
    check_dev_connect()
    conv_param_dict = {'ch_out': '1', 'stride':'[1 1]', 'pad':'[0 0 0 0]', 'kernel':'3x3',
                       'group':'1', 'dilation':'[1 1]', 'flag_bias':'1',
                       'flag_act':'0', 'dtype':'float'}
    batchnorm_param_dict = {'epsilon':'1e-4f', 'momentum':'0.9f',
                            'dtype':'float'}
    pooling_param_dict = {'stride':'2', 'pad':'0', 'kernel':'2x2', 'ceil_mode':'0',
                          'flag_global':'0', 'exclusive':'1', 'pooling_type': 'max',
                          'dtype':'float'}
    activation_param_dict = {'act_type':'relu', 'dtype':'float'}
    fc_param_dict = {'param_dim':'1x1','flag_bias':'1', 'dtype':'float'}
    op_info = {}
    cur_op_name = ''
    cur_param_dict = {}
    input_dims = ''
    output_dims = ''
    runtime_cmd = []
    fid = open(args.ops_path, 'r')
    handle = open(args.latency_lookup_table_path, 'w')
    handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('dev_info'.ljust(30), 'armv7/v8'.ljust(10), 'core_num'.ljust(10), 'thread_num'.ljust(10), 'power_mode'.ljust(10), 'core0 arch'.ljust(10), 'core1 arch'.ljust(10),
                    'core2 arch'.ljust(10), 'core3 arch'.ljust(10), 'core4 arch'.ljust(10), 'core5 arch'.ljust(10),
                    'core6 arch'.ljust(10), 'core7 arch'.ljust(10)))
    dev_info, core_num, arch_type = get_dev_info()
    handle.write('{}\t{}\t{}\t{}'.format(dev_info.ljust(30), str(args.arm_v7_v8).ljust(10), str(core_num).ljust(10), str(args.threads).ljust(10), str(args.power_mode).ljust(10)))
    for i in arch_type:
        handle.write('\t{}'.format(i).ljust(10))
    handle.write('\n')
    handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('op_name'.ljust(10), 'input_dims'.ljust(10), 'output_dims'.ljust(10), 'param_info'.ljust(80), 'min_latency(ms)'.ljust(10), 'max_latency(ms)'.ljust(10), 'avg_latency(ms)'.ljust(10)))
    for line in fid.readlines():
        line = [line.strip('\n')]
        for data_item in line:
            data_item = data_item.strip().split('\t')
            cur_op_name = data_item[0]
            input_dims = data_item[1]
            parameters = data_item[2].strip('( )').split(',')
            for item_ in parameters:
                item_ = item_.strip().split('=')
                # conv op dict
                if cur_op_name == 'conv':
                    cur_param_dict = conv_param_dict
                    if item_[0] == 'ch_out':
                        cur_param_dict['ch_out'] = item_[1]
                    elif item_[0] == 'stride':
                        cur_param_dict['stride'] = item_[1]
                    elif item_[0] == 'pad':
                        cur_param_dict['pad'] = item_[1]
                    elif item_[0] == 'kernel':
                        cur_param_dict['kernel'] = item_[1]
                    elif item_[0] == 'group':
                        cur_param_dict['group'] = item_[1]
                    elif item_[0] == 'dilation':
                        cur_param_dict['dilation'] = item_[1]
                    elif item_[0] == 'flag_bias':
                        cur_param_dict['flag_bias'] = item_[1]
                    elif item_[0] == 'flag_act':
                        cur_param_dict['flag_act'] = item_[1]
                    elif item_[0] == 'dtype':
                        cur_param_dict['dtype'] = item_[1]
                #batchnorm op dict
                elif cur_op_name == 'batchnorm':
                    cur_param_dict = batchnorm_param_dict
                    if item_[0] == 'epsilon':
                        cur_param_dict['epsilon'] = item_[1]
                    elif item_[0] == 'momentum':
                        cur_param_dict['momentum'] = item_[1]
                #pooling op dict
                elif cur_op_name == 'pooling':
                    cur_param_dict = pooling_param_dict
                    if item_[0] == 'stride':
                        cur_param_dict['stride'] = item_[1]
                    elif item_[0] == 'pad':
                        cur_param_dict['pad'] = item_[1]
                    elif item_[0] == 'kernel':
                        cur_param_dict['kernel'] = item_[1]
                    elif item_[0] == 'ceil_mode':
                        cur_param_dict['ceil_mode'] = item_[1]
                    elif item_[0] == 'flag_global':
                        cur_param_dict['flag_global'] = item_[1]
                    elif item_[0] == 'exclusive':
                        cur_param_dict['exclusive'] = item_[1]
                    elif item_[0] == 'pooling_type':
                        cur_param_dict['pooling_type'] = item_[1]
                #activation op dict
                elif cur_op_name == 'activation':
                    cur_param_dict = activation_param_dict
                    if item_[0] == 'act_type':
                        cur_param_dict['act_type'] = item_[1]
                # fc op dict
                elif cur_op_name == 'fc':
                    cur_param_dict = fc_param_dict
                    if item_[0] == 'param_dim':
                        cur_param_dict['param_dim'] = item_[1]
                    elif item_[0] == 'flag_bias':
                        cur_param_dict['flag_bias'] = item_[1]
                    elif item_[0] == 'dtype':
                        cur_param_dict['dtype'] = 'float'
        op_info[cur_op_name] = cur_param_dict

        if cur_op_name == 'conv':
            batch  = input_dims.strip('['  ']').split()[0]
            in_ch  = input_dims.strip('['  ']').split()[1]
            height = input_dims.strip('['  ']').split()[2]
            width  = input_dims.strip('['  ']').split()[3]
            out_ch = cur_param_dict['ch_out']
            pad_top = cur_param_dict['pad'].strip('['  ']').split()[0]
            pad_bottom = cur_param_dict['pad'].strip('['  ']').split()[1]
            pad_left = cur_param_dict['pad'].strip('['  ']').split()[2]
            pad_right = cur_param_dict['pad'].strip('['  ']').split()[0]
            dila_h = cur_param_dict['dilation'].strip('['  ']').split()[0]
            dila_w = cur_param_dict['dilation'].strip('['  ']').split()[1]
            kernel_h = cur_param_dict['kernel'][0]
            kernel_w = cur_param_dict['kernel'][2]
            stride_h = cur_param_dict['stride'].strip('['  ']').split()[0]
            stride_w = cur_param_dict['stride'].strip('['  ']').split()[1]
            hout = (int(height) + int(pad_top) + int(pad_bottom) - int(dila_h) * 
                   (int(kernel_h) - 1) + 1) / int(stride_h) + 1
            wout = (int(width) + int(pad_left) + int(pad_right) - int(dila_w) * 
                   (int(kernel_w) - 1) + 1) / int(stride_w) + 1
            output_dims = '[' +  str(batch) + ' ' + str(out_ch) + ' ' + str(int(hout)) + ' ' + str(int(wout)) + ']'
            dtype = 0
            if cur_param_dict['dtype'] == 'float':
                dtype = 0
            elif cur_param_dict['dtype'] == 'int8_float':
                dtype = 1
            elif cur_param_dict['dtype'] == 'int8_int8':
                dtype = 2
            runtime_cmd = [str(batch), str(in_ch), str(height), str(width), str(out_ch),
                           str(cur_param_dict['group']), str(cur_param_dict['kernel'])[0],
                           str(pad_top), str(pad_bottom),
                           str(pad_left), str(pad_right),
                           str(stride_h), str(stride_w),
                           str(dila_h), str(dila_w),
                           str(cur_param_dict['flag_bias']), str(cur_param_dict['flag_act']),
                           str(dtype)]
        elif cur_op_name == 'batchnorm':
            batch  = input_dims.strip('['  ']').split()[0]
            in_ch  = input_dims.strip('['  ']').split()[1]
            height = input_dims.strip('['  ']').split()[2]
            width  = input_dims.strip('['  ']').split()[3]
            output_dims = input_dims
            runtime_cmd = [str(batch), str(in_ch), str(height), str(width),
                           str(cur_param_dict['epsilon']), str(cur_param_dict['momentum'])]
        elif cur_op_name == 'pooling':
            batch  = input_dims.strip('['  ']').split()[0]
            in_ch  = input_dims.strip('['  ']').split()[1]
            height = input_dims.strip('['  ']').split()[2]
            width  = input_dims.strip('['  ']').split()[3]
            hout   = 1
            wout   = 1
            pad_top = cur_param_dict['pad'].strip('['  ']').split()[0]
            pad_bottom = cur_param_dict['pad'].strip('['  ']').split()[1]
            pad_left = cur_param_dict['pad'].strip('['  ']').split()[2]
            pad_right = cur_param_dict['pad'].strip('['  ']').split()[3]
            kernel_h = cur_param_dict['kernel'][0]
            kernel_w = cur_param_dict['kernel'][2]
            stride_h = cur_param_dict['stride'].strip('['  ']').split()[0]
            stride_w = cur_param_dict['stride'].strip('['  ']').split()[1]
            if cur_param_dict['flag_global'] == '0':
                if cur_param_dict['ceil_mode'] == '0':
                    hout = (int(height) - int(kernel_h) + int(pad_top) + int(pad_bottom)) / int(stride_h) + 1
                    wout = (int(width) - int(kernel_w) + int(pad_left) + int(pad_right)) / int(stride_w) + 1
                else:
                    hout = (int(height) - int(kernel_h) + int(pad_top) + int(pad_bottom) + int(stride_h) - 1) / int(stride_h) + 1
                    wout = (int(width) - int(kernel_w) + int(pad_left) + int(pad_right) + int(stride_w) - 1) / int(stride_w) + 1
            output_dims = '[' + batch + ' ' + str(in_ch) + ' ' + str(int(hout)) + ' ' + str(int(wout)) + ']'
            pooling_type = 0
            if cur_param_dict['pooling_type'] == 'max':
                pooling_type = 0
            else:
                pooling_type = 1
            runtime_cmd = [str(batch), str(in_ch), str(height), str(width),
                           str(stride_h), str(stride_w),
                           str(pad_top), str(pad_bottom),
                           str(pad_left), str(pad_right),
                           str(cur_param_dict['kernel'])[0], str(cur_param_dict['ceil_mode']),
                           str(cur_param_dict['flag_global']), str(cur_param_dict['exclusive']),
                           str(pooling_type)]
        elif cur_op_name == 'activation':
            batch  = input_dims.strip('['  ']').split()[0]
            in_ch  = input_dims.strip('['  ']').split()[1]
            height = input_dims.strip('['  ']').split()[2]
            width  = input_dims.strip('['  ']').split()[3]
            act_type = 1
            if cur_param_dict['act_type'] == 'relu':
                act_type = 1
            elif cur_param_dict['act_type'] == 'relu6':
                act_type = 2
            elif cur_param_dict['act_type'] == 'leaky_relu':
                act_type = 4
            elif cur_param_dict['act_type'] == 'sigmoid':
                act_type = 5
            elif cur_param_dict['act_type'] == 'tanh':
                act_type = 6
            elif cur_param_dict['act_type'] == 'swish':
                act_type = 7
            elif cur_param_dict['act_type'] == 'exp':
                act_type = 8
            elif cur_param_dict['act_type'] == 'abs':
                act_type = 9
            elif cur_param_dict['act_type'] == 'hard_swish':
                act_type = 10
            elif cur_param_dict['act_type'] == 'reciprocal':
                act_type = 11
            elif cur_param_dict['act_type'] == 'threshold_relu':
                act_type = 12
            output_dims = input_dims
            runtime_cmd = [str(batch), str(in_ch), str(height), str(width),
                           str(act_type)]
        elif cur_op_name == 'fc':
            m = input_dims.strip('['  ']').split()[0]
            k = input_dims.strip('['  ']').split()[1]
            n = cur_param_dict['param_dim'].split('x')[1]
            output_dims = '[' + m + ' ' + n + ']'
            runtime_cmd = [str(m), str(n), str(k), str(cur_param_dict['flag_bias']),
                           str(cur_param_dict['dtype'])]

        avg_latency, min_latency, max_latency = get_op_latency([cur_op_name] +
                                 runtime_cmd + [str(args.threads), str(args.power_mode),
                                 str(args.warmup_times), str(args.repeats_times)],
                                 args.platform)

        param_dict = ''
        for k in cur_param_dict:
            param_dict += str(k) + '=' + str(cur_param_dict[k]) + ','
        param_dict = '(' + param_dict[:-1] + ')'
        handle.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(cur_op_name.ljust(10), input_dims.ljust(10), output_dims.ljust(10), param_dict.ljust(80), str(min_latency).ljust(10), str(max_latency).ljust(10), str(avg_latency).ljust(10)))

    fid.close()
    handle.close()
    print('Congratulations! Get Latency LookUp Table is Completed.')

if __name__ == '__main__':
    main()
