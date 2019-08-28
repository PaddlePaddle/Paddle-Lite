# -*- coding: utf-8 -*
import os
import sys
import math
import subprocess
import numpy as np
import paddle.fluid as fluid

model_path = "yolov2"
checked_model_path = "checked_model"
feed_path = "feeds"
output_path = "outputs"
diff_threshold = 0.05
is_lod = False
mobile_model_path = ""
fast_check = False
is_sample_step = False
sample_step = 1
sample_num = 20
need_encrypt = False
checked_encrypt_model_path = "checked_encrypt_model"
output_var_filter = []
output_key_filter = {}
check_shape = False

np.set_printoptions(linewidth=150)

mobile_exec_root = "/data/local/tmp/bin"
mobile_src_root = os.path.abspath("../../../")
if mobile_src_root.endswith("/"):
    mobile_src_root = mobile_src_root[:-1]

dot = "•"
black = lambda x: "\033[30m" + str(x) + "\033[0m"
red = lambda x: "\033[31m" + str(x) + "\033[0m"
green = lambda x: "\033[32m" + str(x) + "\033[0m"
yellow = lambda x: "\033[33m" + str(x) + "\033[0m"
reset = lambda x: "\033[0m" + str(x)

def pp_tab(x, level=0):
    header = ""
    for i in range(0, level):
        header += "\t"
    print(header + str(x))
def pp_black(x, level=0):
    pp_tab(black(x) + reset(""), level)
def pp_red(x, level=0):
    pp_tab(red(x) + reset(""), level)
def pp_green(x, level=0):
    pp_tab(green(x) + reset(""), level)
def pp_yellow(x, level=0):
    pp_tab(yellow(x) + reset(""), level)

def sh(command):
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return pipe.stdout.read().decode("utf-8")
def push(src, dest=""):
    sh("adb push {} {}".format(src, mobile_exec_root + "/" + dest))

pp_yellow(dot + " start inspecting fluid model")

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

# 加载模型
def load_model(model_path):
    prog, feeds, fetches = fluid.io.load_inference_model(dirname=model_path, executor=exe, model_filename="model", params_filename="params")
    return (prog, feeds, fetches)

prog, feeds, fetches = load_model(model_path)

# 强制要求所有张量的形状，在model和params中一致，并重新保存模型
def resave_model(feed_kv):
    if len(mobile_model_path) > 0:
        pp_green("has set mobile_model_path, stop checking model & params", 1)
        sh("cp {}/* {}".format(mobile_model_path, checked_model_path))
        return
    ops = prog.current_block().ops
    vars = prog.current_block().vars
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
    fluid.io.save_inference_model(dirname=checked_model_path, feeded_var_names=feeds, target_vars=fetches, executor=exe, main_program=prog, model_filename="model", params_filename="params")
    if has_found_wrong_shape:
        pp_red("has found wrong shape", 1)
    else:
        pp_green("has not found wrong shape", 1)
    pp_green("new model is saved into directory 【{}】".format(checked_model_path), 1)

# 分别加密model和params，加密key使用同一个
def encrypt_model():
    if not need_encrypt:
        return
    pp_yellow(dot + dot + " encrypting model")
    if not os.path.exists(checked_encrypt_model_path):
        os.mkdir(checked_encrypt_model_path)
    res = sh("model-encrypt-tool/enc_key_gen -l 20 -c 232")
    lines = res.split("\n")

    for line in lines:
        if line.startswith("key:"):
            line = line.replace('key:','')
            sh("model-encrypt-tool/enc_model_gen -k '{}' -c 2 -i checked_model/model -o "
               "checked_model/model.ml".format(line))
            sh("model-encrypt-tool/enc_model_gen -k '{}' -c 2 -i checked_model/params  -o checked_model/params.ml".format(line))
            pp_green("model has been encrypted, key is : {}".format(line), 1)
            sh("mv {} {}".format(checked_model_path + "/*.ml", checked_encrypt_model_path))
            return
    pp_red("model encrypt error", 1)

# 生成feed的key-value对
def gen_feed_kv():
    feed_kv = {}
    for feed_name in feeds:
        feed_shape = get_feed_var_shape(feed_name)
        data = np.random.random(feed_shape).astype("float32")
        feed_kv[feed_name] = data
    return feed_kv

# 保存feed的key-value对
def save_feed_kv(feed_kv):
    for feed_name in feed_kv:
        feed_data = feed_kv[feed_name]
        feed_list = feed_data.flatten().tolist()
        if not os.path.exists(feed_path):
            os.mkdir(feed_path)
        file_name = feed_name.replace("/", "_")
        out_file = open(feed_path + "/" + file_name, "w")
        for feed_item in feed_list:
            out_file.write("{}\n".format(feed_item))
        out_file.close()

last_feed_var_name = None
last_feed_file_name = None
last_feed_var_lod = None
# 加载feed的key-value对
def load_feed_kv():
    if not os.path.exists(feed_path):
        return None
    global last_feed_var_name
    global last_feed_file_name
    global last_feed_var_lod
    feed_kv = {}
    pp_yellow(dot + dot + " checking feed info")
    pp_green("feed data is saved into directory 【{}】".format(feed_path), 1)
    for feed_name in feeds:
        feed_shape = get_feed_var_shape(feed_name)
        pp_tab("feed var name : {}; feed var shape : {}".format(feed_name, feed_shape), 1)
        file_name = feed_name.replace("/", "_")
        last_feed_var_name = feed_name
        last_feed_file_name = file_name
        feed_file_path = feed_path + "/" + file_name
        if not os.path.exists(feed_file_path):
            return None
        data = np.loadtxt(feed_file_path)
        expected_len = 1
        for dim in feed_shape:
            expected_len *= dim
        if len(np.atleast_1d(data)) != expected_len:
            return None
        data = data.reshape(feed_shape).astype("float32")
        
        if is_lod:
            data_shape = [1]
            for dim in feed_shape:
                data_shape.append(dim)
            data = data.reshape(data_shape).astype("float32")
            tensor = fluid.LoDTensor()
            seq_lens = [len(seq) for seq in data]
            cur_len = 0
            lod = [cur_len]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            data = data.reshape(feed_shape)
            tensor.set(data, fluid.CPUPlace())
            tensor.set_lod([lod])
            last_feed_var_lod = lod
            feed_kv[feed_name] = tensor
        else:
            feed_kv[feed_name] = data
    return feed_kv

# 运行模型
def run_model(feed_kv=None):
    if feed_kv is None:
        feed_kv = gen_feed_kv()
    outputs = exe.run(prog, feed=feed_kv, fetch_list=fetches, return_numpy=False)
    results = []
    for output in outputs:
        results.append(np.array(output))
    return results

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

persistable_cache = []
# 所有var，全部变成持久化
def force_all_vars_to_persistable():
    global persistable_cache
    for var_name in vars.keys():
        var_name = str(var_name)
        v = fluid.framework._get_var(var_name, prog)
        persistable = v.persistable
        if not persistable:
            persistable_cache.append(var_name)
            v.persistable = True

# 恢复持久化属性
def restore_all_vars_persistable():
    global persistable_cache
    for var_name in vars.keys():
        var_name = str(var_name)
        v = fluid.framework._get_var(var_name, prog)
        persistable = v.persistable
        if var_name in persistable_cache:
            v.persistable = False
    persistable_cache = []

# 获取var的数据
def get_var_data(var_name, feed_kv=None):
    output = np.array(fluid.global_scope().var(var_name).get_tensor())
    return output

output_var_cache = {}
def tensor_sample(tensor):
    if is_sample_step:
        step = sample_step
    else:
        step = math.floor(len(tensor) / sample_num)
    step = max(step, 1)
    step = int(step)
    sample = []
    for i in range(0, len(tensor), step):
        sample.append(tensor[i])
    return sample

op_cache = {}
# 获取每层输出的数据
def save_all_op_output(feed_kv=None):
    force_all_vars_to_persistable()
    outputs = run_model(feed_kv=feed_kv)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ops = prog.current_block().ops
    fetch_names = []
    for fetch in fetches:
        fetch_names.append(fetch.name)
    feed_names = feeds
    for fetch_name in fetch_names:
        output_var_filter.append(fetch_name)
    for i in range(len(ops)):
        op = ops[i]
        var_name = None
        var_name_index = -1
        for index in range(len(op.output_names)):
            if op.output_names[index] in ["Y", "Out", "Output"]:
                var_name_index = index
                break
        if var_name_index != -1:
            var_name = op.output_arg_names[var_name_index]
        else:
            for name in op.output_arg_names:
                var_name = name
                if "tmp" in name:
                    break
        if len(output_var_filter) > 0:
            if var_name not in output_var_filter:
                continue
        # real_var_name = None
        # if op.type == "fetch":
        #     for name in op.input_arg_names:
        #         real_var_name = name
        #         if "tmp" in name:
        #             break
        # else:
        #     real_var_name = var_name
        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            sample = tensor_sample(data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            out_file = open(output_path + "/" + file_name, "w")
            if var_name in feed_names:
                for item in data:
                    out_file.write("{}\n".format(item))
            else:
                for item in sample:
                    out_file.write("{}\n".format(item))
            out_file.close()
        except:
            pass
    for i in range(len(ops)):
        op = ops[i]
        if op.type not in output_key_filter:
            continue
        var_name = None
        var_name_index = -1
        for index in range(len(op.output_names)):
            if op.output_names[index] in output_key_filter[op.type]:
                var_name_index = index
                break
        if var_name_index != -1:
            var_name = op.output_arg_names[var_name_index]
        else:
            continue
        if len(output_var_filter) > 0:
            if var_name not in output_var_filter:
                continue
        # real_var_name = None
        # if op.type == "fetch":
        #     for name in op.input_arg_names:
        #         real_var_name = name
        #         if "tmp" in name:
        #             break
        # else:
        #     real_var_name = var_name
        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            sample = tensor_sample(data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            out_file = open(output_path + "/" + file_name, "w")
            if var_name in feed_names:
                for item in data:
                    out_file.write("{}\n".format(item))
            else:
                for item in sample:
                    out_file.write("{}\n".format(item))
            out_file.close()
        except:
            pass
    pp_green("all the op outputs are saved into directory 【{}】".format(output_path), 1)
    restore_all_vars_persistable()

ops = prog.current_block().ops
vars = prog.current_block().vars

pp_yellow(dot + dot + " checking op list")
op_types = set()
for op in ops:
    op_types.add(op.type)
pp_tab("op types : {}".format(op_types), 1)

def check_mobile_results(args, fuse, mem_opt):
    args = "{} {} {}".format("1" if fuse else "0", "1" if mem_opt else "0", args)
    res = sh("adb shell \"cd {} && export LD_LIBRARY_PATH=. && ./test-net {}\"".format(mobile_exec_root, args))
    lines = res.split("\n")
    for line in lines:
        print(line)
    for line in lines:
        if line.startswith("auto-test-debug"):
            print(line)
    pp_yellow(dot + dot + " checking paddle mobile results for {} -- {} ".format(green("【fusion】" if fuse else "【non fusion】"), green("【memory-optimization】" if mem_opt else "【non-memory-optimization】")))
    mobile_var_cache = {}
    for line in lines:
        parts = line.split(" ")
        if len(parts) < 2:
            continue
        if "auto-test" != parts[0]:
            continue
        if parts[1] == "load-time-cost":
            pp_green("load time cost : {}".format(parts[2]), 1) 
        elif parts[1] == "predict-time-cost":
            pp_green("predict time cost : {}".format(parts[2]), 1) 
        elif parts[1] == "preprocess-time-cost":
            pp_green("preprocess time cost : {}".format(parts[2]), 1)
        elif parts[1] == "var":
            var_name = parts[2]
            values = list(map(lambda x: float(x), parts[3:]))
            mobile_var_cache[var_name] = values
    error_index = None
    error_values1 = None
    error_values2 = None
    checked_names = []
    fetch_names = []
    for fetch in fetches:
        fetch_names.append(fetch.name)
    for index in op_cache:
        op_output_var_name, op = op_cache[index]
        if mem_opt:
            found_in_fetch = False
            for fetch in fetches:
                if op_output_var_name == fetch.name:
                    found_in_fetch = True
                    break
            if not found_in_fetch:
                continue
        if not op_output_var_name in output_var_cache:
            continue
        if not op_output_var_name in mobile_var_cache:
            continue
        values1 = output_var_cache[op_output_var_name]
        values2 = mobile_var_cache[op_output_var_name]
        shape = get_var_shape(op_output_var_name) if check_shape else []
        if len(values1) + len(shape) != len(values2):
            error_index = index
        for i in range(len(shape)):
            v1 = shape[i]
            v2 = values2[i]
            if v1 != v2:
                error_index = index
                break
        if error_index == None:
            for i in range(len(values1)):
                v1 = values1[i]
                v2 = values2[len(shape) + i]
                if abs(v1 - v2) > diff_threshold:
                    error_index = index
                    break
        checked_names.append(op_output_var_name)
        if error_index != None:
            error_values1 = values1
            error_values2 = values2
            break
    if error_index == None:
        for name in fetch_names:
            if name not in checked_names:
                error_index = -1
                break
    if error_index == None:
        pp_green("outputs are all correct", 1)
    elif error_index == -1:
        pp_red("outputs are missing")
    else:
        error_values1 = np.array(error_values1)
        error_values2 = np.array(error_values2)
        # pp_red("mobile op is not correct, error occurs at {}th op, op's type is {}")
        pp_red("corresponding fluid op is {}th op, op's type is {}, wrong var name is {}".format(
            error_index,op_cache[error_index][1].type,op_output_var_name), 1)
        pp_red("fluid results are : ", 1)
        pp_red(str(error_values1).replace("\n", "\n" + "\t" * 1), 1)
        pp_yellow("paddle mobile results are : ", 1)
        pp_red(str(error_values2).replace("\n", "\n" + "\t" * 1), 1)
    # print(output_var_cache)
    # print(mobile_var_cache)

def main():
    # 加载kv
    feed_kv = load_feed_kv()
    if feed_kv == None:
        feed_kv = gen_feed_kv()
        save_feed_kv(feed_kv)
        feed_kv = load_feed_kv()
    # 预测
    pp_yellow(dot + dot + " checking inference")
    outputs = run_model(feed_kv=feed_kv)
    pp_tab("fluid output : {}".format(outputs), 1)
    # 重新保存模型
    pp_yellow(dot + dot + " checking model correctness")
    resave_model(feed_kv=feed_kv)
    # 输出加密模型
    encrypt_model()
    # 输出所有中间结果
    pp_yellow(dot + dot + " checking output result of every op")
    save_all_op_output(feed_kv=feed_kv)
    pp_yellow(dot + dot + " checking fetch info")
    for fetch in fetches:
        fetch_name = fetch.name
        fetch_shape = get_var_shape(fetch_name)
        pp_tab("fetch var name : {}; fetch var shape : {}".format(fetch_name, fetch_shape), 1)
    # 输出所有op、var信息
    info_file = open("info.txt", "w")
    for i in range(len(ops)):
        op = ops[i]
        info_file.write("{}th op: type - {}\n".format(i, op.type))
        info_file.write("inputs:\n")
        for var_name in op.input_arg_names:
            try:
                shape = get_var_shape(var_name)
                shape_str = ", ".join(list(map(lambda x: str(x), shape)))
                info_file.write("var {} : {}\n".format(var_name, shape_str))
            except:
                pass
        info_file.write("outputs:\n")
        for var_name in op.output_arg_names:
            try:
                shape = get_var_shape(var_name)
                shape_str = ", ".join(list(map(lambda x: str(x), shape)))
                info_file.write("var {} : {}\n".format(var_name, shape_str))
            except:
                pass
    info_file.close()
    # 开始检查mobile的正确性
    print("")
    print("==================================================")
    print("")
    pp_yellow(dot + " start inspecting paddle mobile correctness & performance")
    push(checked_model_path)
    push(feed_path + "/" + last_feed_file_name, "input.txt")
    push(mobile_src_root + "/build/release/arm-v7a/build/libpaddle-mobile.so")
    push(mobile_src_root + "/build/release/arm-v7a/build/cl_kernel")
    push(mobile_src_root + "/test/build/test-wrap")
    res = sh("adb shell 'cd {} && export LD_LIBRARY_PATH=. && ./test-wrap'".format(mobile_exec_root))
    lines = res.split("\n")
    for line in lines:
        print(line)

if __name__ == "__main__":
    main()
