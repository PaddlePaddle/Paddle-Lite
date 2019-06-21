import os
import sys
import math
import subprocess
import numpy as np
import paddle.fluid as fluid

model_path = "model"
checked_model_path = "checked_model"
feed_path = "feeds"
output_path = "outputs"

mobile_exec_root = "/data/local/tmp/bin"
mobile_src_root = os.path.abspath("../../../")
if mobile_src_root.endswith("/"):
    mobile_src_root = mobile_src_root[:-1]

dot = "•"
black = lambda x: "\033[30m" + str(x)
red = lambda x: "\033[31m" + str(x)
green = lambda x: "\033[32m" + str(x)
reset = lambda x: "\033[0m" + str(x)
yellow = lambda x: "\033[33m" + str(x)

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
def resave_model():
    ops = prog.current_block().ops
    vars = prog.current_block().vars
    # 强制所有var为可持久化
    p_names = []
    for name in vars:
        v = fluid.framework._get_var(name, prog)
        if not v.persistable:
            v.persistable = True
            p_names.append(name)
    outputs = run_model()
    has_found_wrong_shape = False
    # 修正每个var的形状
    for name in vars:
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

# 生成feed的key-value对
def gen_feed_kv():
    feed_kv = {}
    for feed_name in feeds:
        feed_shape = get_var_shape(feed_name)
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
# 加载feed的key-value对
def load_feed_kv():
    global last_feed_var_name
    global last_feed_file_name
    feed_kv = {}
    pp_yellow(dot + dot + " checking feed info")
    pp_green("feed data is saved into directory 【{}】".format(feed_path), 1)
    for feed_name in feeds:
        feed_shape = get_var_shape(feed_name)
        pp_tab("feed var name : {}; feed var shape : {}".format(feed_name, feed_shape), 1)
        file_name = feed_name.replace("/", "_")
        last_feed_var_name = feed_name
        last_feed_file_name = file_name
        data = np.loadtxt(feed_path + "/" + file_name).reshape(feed_shape).astype("float32")
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

# 获取var的数据
def get_var_data(var_name, feed_kv=None):
    # 强制var为可持久化
    v = fluid.framework._get_var(var_name, prog)
    persistable = v.persistable
    if not persistable:
        v.persistable = True
    outputs = run_model(feed_kv=feed_kv)
    output = np.array(fluid.global_scope().find_var(var_name).get_tensor())
    # 恢复var的可持久化属性
    v.persistable = persistable
    return output

output_var_cache = {}
def tensor_sample(tensor):
    step = math.floor(len(tensor) / 20)
    sample = []
    for i in range(0, len(tensor), step):
        sample.append(tensor[i])
    return sample
op_cache = {}

# 获取每层输出的数据
def save_all_op_output(feed_kv=None):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ops = prog.current_block().ops
    for i in range(len(ops)):
        op = ops[i]
        var_name = None
        for name in op.output_arg_names:
            var_name = name
            if "tmp" in name:
                break
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            sample = tensor_sample(data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            out_file = open(output_path + "/" + file_name, "w")
            for item in data:
                out_file.write("{}\n".format(item))
            out_file.close()
        except:
            pass
    pp_green("all the op outputs are saved into directory 【{}】".format(output_path), 1)

ops = prog.current_block().ops
vars = prog.current_block().vars

pp_yellow(dot + dot + " checking op list")
op_types = set()
for op in ops:
    op_types.add(op.type)
pp_tab("op types : {}".format(op_types), 1)

def check_mobile_results(lines, fuse):
    pp_yellow(dot + dot + " checking {} paddle mobile results".format("fusion" if fuse else "non fusion"))
    mobile_var_cache = {}
    for line in lines:
        parts = line.split(" ")
        if len(parts) <= 0:
            continue
        if fuse:
            if "auto-test-fuse" != parts[0]:
                continue
        else:
            if "auto-test" != parts[0]:
                continue
        if parts[1] == "load-time-cost":
            pp_green("load time cost : {}".format(parts[2]), 1) 
        elif parts[1] == "predict-time-cost":
            pp_green("predict time cost : {}".format(parts[2]), 1) 
        elif parts[1] == "var":
            var_name = parts[2]
            values = list(map(lambda x: float(x), parts[3:]))
            mobile_var_cache[var_name] = values
    error_index = None
    error_values1 = None
    error_values2 = None
    for index in op_cache:
        op_output_var_name, op = op_cache[index]
        if not op_output_var_name in output_var_cache:
            continue
        if not op_output_var_name in mobile_var_cache:
            continue
        values1 = output_var_cache[op_output_var_name]
        values2 = mobile_var_cache[op_output_var_name]
        if len(values1) != len(values2):
            error_index = index
        if error_index == None:
            for i in range(len(values1)):
                v1 = values1[i]
                v2 = values2[i]
                if abs(v1 - v2) > 0.01:
                    error_index = index
                    break
        if error_index != None:
            error_values1 = values1
            error_values2 = values2
            break
    if error_index == None:
        pp_green("outputs are all correct", 1)
    else:
        pp_red("{} op's output is not correct, op's type is {}".format(error_index, op_cache[error_index][1].type), 1)
        pp_red("fluid results are : {}".format(error_values1), 1)
        pp_red("paddle mobile results are : {}".format(error_values2), 1)
    # print(output_var_cache)
    # print(mobile_var_cache)

def main():
    # 如果feed_path不存在，则需要生成并保存feed的键值对
    if not os.path.exists(feed_path):
        feed_kv = gen_feed_kv()
        save_feed_kv(feed_kv)
    # 加载kv
    feed_kv = load_feed_kv()
    pp_yellow(dot + dot + " checking fetch info")
    for fetch in fetches:
        pp_tab("fetch var name : {}".format(fetch.name), 1)
    # 预测
    pp_yellow(dot + dot + " checking inference")
    outputs = run_model(feed_kv=feed_kv)
    pp_tab("fluid output : {}".format(outputs), 1)
    # 重新保存模型
    pp_yellow(dot + dot + " checking model correctness")
    resave_model()
    # 输出所有中间结果
    pp_yellow(dot + dot + " checking output result of every op")
    save_all_op_output(feed_kv=feed_kv)
    # 开始检查mobile的正确性
    print("")
    print("==================================================")
    print("")
    pp_yellow(dot + " start inspecting paddle mobile correctness & performance")
    push(checked_model_path)
    push(feed_path + "/" + last_feed_file_name, "input.txt")
    push(mobile_src_root + "/build/release/arm-v7a/build/libpaddle-mobile.so")
    push(mobile_src_root + "/test/build/test-net")
    last_feed_var_shape = get_var_shape(last_feed_var_name)
    args = str(len(last_feed_var_shape))
    for dim in last_feed_var_shape:
        args += " " + str(dim)
    args += " " + str(len(output_var_cache))
    for var_name in output_var_cache.keys():
        args += " " + var_name
    res = sh("adb shell \"cd {} && export LD_LIBRARY_PATH=. && ./test-net {}\"".format(mobile_exec_root, args))
    lines = res.split("\n")
    check_mobile_results(lines, False)
    check_mobile_results(lines, True)

if __name__ == "__main__":
    main()
