# -*- coding: utf-8 -*
import os
import sys
import math
import qrcode
import subprocess
import numpy as np
import paddle.fluid as fluid
from flask import Flask, request, send_from_directory, jsonify, make_response

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from fluidtools import run
from fluidtools import check_model

dump_data_and_model = False

def get_ip_address():
    handle = os.popen("ifconfig | grep 172 | grep inet | grep netmask | grep broadcast | cut -d \" \" -f2")
    ip = handle.read()
    ip = ip.strip()
    return ip

app = Flask(__name__, static_url_path='')

param_precisions = [1] # 0 for float16, 1 for float32

def process_model(precision, name):
    model_dir = "./{}/{}".format(precision, name)
    os.chdir(model_dir)
    os.chdir("../..")
    var_info = check_model(model_dir, dump_data_and_model)
    return var_info

def get_model_info(precision, name):
    # model_info = {
    #     "name": name,
    #     "params_precision": [precision],
    #     "fusion": [True, False],
    #     "reuse_texture": [True, False],
    #     "use_mps": [True, False],
    #     "test_performance": True,
    #     "diff_precision": 0.01,
    #     "vars_dic": {
    #     }
    # }
    model_info = {
        "name": name,
        "params_precision": [precision],
        "fusion": [True],
        "reuse_texture": [True],
        "use_mps": [True, False],
        "test_performance": False,
        "diff_precision": 0.01,
        "vars_dic": {
        }
    }
    var_info = process_model(precision, name)
    model_info["vars_dic"] = var_info
    return model_info

model_list = []
def process_models():
    for precision in param_precisions:
        model_names = os.listdir("./{}".format(precision))
        for name in model_names:
            model_info = get_model_info(precision, name)
            model_list.append(model_info)

@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory('images', path)

@app.route('/getFile/<name>/model')
def send_model(name):
    precision = 1
    return send_from_directory("{}/{}".format(precision, name), "model-checked")

@app.route('/getFile/<name>/params/<precision>')
def send_params(name, precision):
    return send_from_directory("{}/{}".format(precision, name), "params-checked")

@app.route('/getFile/<name>/data/<var>')
def send_data(name, var):
    precision = 1
    return send_from_directory("{}/{}/data".format(precision, name), var)

@app.route('/getTestInfo', methods=['GET'])
def test_info():
    info = {"model_list": model_list}
    return make_response(jsonify(info), 200)

test_result = None
@app.route('/putTestResult', methods=['POST'])
def put_test_result():
    global test_result
    test_result = request.get_json()
    success = True
    for item in test_result["results"]:
        result = item["isResultEqual"]
        if not result:
            success = False
            break
    test_result["aaa-success"] = success
    os.popen("open -a \"/Applications/Google Chrome.app\" \"{}/showTestResult\"".format(host))
    return make_response(jsonify({"msg": "ok"}), 200)

@app.route('/showTestResult', methods=['GET'])
def show_test_result():
    global test_result
    return make_response(jsonify(test_result), 200)

@app.route('/', methods=['GET'])
def home():
    return "<html><body><img src=\"images/qrcode.png\"/></body></html>"

host = None

if __name__ == "__main__":
    process_models()
    host = "http://{}:8080".format(get_ip_address())
    image = qrcode.make(host)
    if not os.path.isdir("images"):
        os.mkdir("images")
    image.save("images/qrcode.png")
    os.popen("open -a \"/Applications/Google Chrome.app\" \"{}\"".format(host))
    app.run(host="0.0.0.0", port=8080)

