# -*- coding: utf-8 -*
import os
import sys
import math
import struct
import subprocess
import numpy as np
import paddle.fluid as fluid

fast_check = False

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

ops = None
def check_model(model_path, dump_data_and_model):
    check_model_impl(model_path, dump_data_and_model, True)
    return check_model_impl(model_path, dump_data_and_model, False)

def check_model_impl(model_path, dump_data_and_model, need_check):
    global ops
    if need_check:
        prog, feeds, fetches = fluid.io.load_inference_model(dirname=model_path, executor=exe, model_filename="model", params_filename="params")
    else:
        prog, feeds, fetches = fluid.io.load_inference_model(dirname=model_path, executor=exe, model_filename="model-checked", params_filename="params-checked")
    ops = prog.current_block().ops
    vars = prog.current_block().vars
    
    # 获取变量形状
    def get_var_shape(var_name):
        vars = prog.current_block().vars
        shape = vars[var_name].desc.shape()
        for i in range(len(shape)):
            dim = shape[i]
            if dim == -1:
                shape[i] = 1
        return shape
    
    # 获取输入变量形状
    def get_feed_var_shape(var_name):
        # 如果想写死输入形状，放开以下语句
        # return [1, 3, 224, 224]
        return get_var_shape(var_name)

    # 生成feed的key-value对
    def gen_feed_kv():
        feed_kv = {}
        for feed_name in feeds:
            feed_shape = get_feed_var_shape(feed_name)
            data = np.random.random(feed_shape).astype("float32")
            feed_kv[feed_name] = data
        return feed_kv

    feed_kv = gen_feed_kv()

    # 运行模型
    def run_model(feed_kv=None):
        if feed_kv is None:
            feed_kv = gen_feed_kv()
        outputs = exe.run(prog, feed=feed_kv, fetch_list=fetches, return_numpy=False)
        results = []
        for output in outputs:
            results.append(np.array(output))
        return results

    # 获取var的数据
    def get_var_data(var_name, feed_kv=None):
        # 强制var为可持久化
        v = fluid.framework._get_var(var_name, prog)
        persistable = v.persistable
        if not persistable:
            v.persistable = True
        # outputs = run_model(feed_kv=feed_kv)
        output = np.array(fluid.global_scope().find_var(var_name).get_tensor())
        # 恢复var的可持久化属性
        v.persistable = persistable
        return output

    # 强制所有var为可持久化
    p_names = []
    for name in vars:
        name = str(name)
        v = fluid.framework._get_var(name, prog)
        if not v.persistable:
            v.persistable = True
            p_names.append(name)
    outputs = run_model(feed_kv=feed_kv)
    has_found_wrong_shape = False
    # 修正每个var的形状
    for name in vars:
        name = str(name)
        v = vars[name]
        if v.persistable:
            v1 = fluid.global_scope().find_var(name)
            try:
                t1 = v1.get_tensor()
                shape = t1.shape()
            except:
                continue
            if v.desc.shape() != shape:
                has_found_wrong_shape = True
            v.desc.set_shape(shape)
    # 恢复var的可持久化属性
    for name in p_names:
        v = fluid.framework._get_var(name, prog)
        v.persistable = False
    if need_check and dump_data_and_model:
        fluid.io.save_inference_model(dirname=model_path, feeded_var_names=feeds, target_vars=fetches, executor=exe, main_program=prog, model_filename="model-checked", params_filename="params-checked")
        return
    var_cache = {}
    # 获取每层输出的数据
    def save_all_op_output(feed_kv=None):
        output_path = "{}/data".format(model_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        ops = prog.current_block().ops
        fetch_names = []
        for fetch in fetches:
            fetch_names.append(fetch.name)
        feed_names = feeds
        for i in range(len(ops)):
            op = ops[i]
            var_name = None
            for name in op.output_arg_names:
                var_name = name
                if "tmp" in name:
                    break
            real_var_name = None
            if op.type == "fetch":
                for name in op.input_arg_names:
                    real_var_name = name
                    if "tmp" in name:
                        break
            else:
                real_var_name = var_name
            if fast_check:
                if var_name not in fetch_names and var_name not in feed_names:
                    continue
            try:
                shape = get_var_shape(var_name)
                var_cache[var_name] = shape
            except:
                pass
            if not dump_data_and_model:
                continue
            try:
                np_data = get_var_data(real_var_name, feed_kv=feed_kv)
                index = -1
                for i in range(len(fetch_names)):
                    if real_var_name == fetch_names[i]:
                        index = i
                        break
                if index != -1:
                    np_data = outputs[index]
                data = np_data.flatten().tolist()
                file_name = var_name.replace("/", "_")
                var_path = output_path + "/" + file_name
                np_data.tofile(var_path)
                # out_file = open(var_path, "wb")
                # if var_name in feed_names:
                #     for item in data:
                #         out_file.write(struct.pack("d", item))
                # else:
                #     for item in data:
                #         out_file.write(struct.pack("d", item))
                # out_file.close()
            except:
                print("dump {} {} failed".format(op.type, var_name))
                pass
    save_all_op_output()
    return var_cache

if __name__ == "__main__":
    model_path = "./1/mobilenet"
    check_model(model_path, True)
