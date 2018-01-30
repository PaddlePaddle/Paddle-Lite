/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include <stdio.h>
#include <limits.h>
#include <limits>
#include <fstream>
#include <set>
#include <iostream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <sstream>
#include <math.h>
#include "caffe.pb.h"

using std::min;
using std::max;
using std::map;
using std::cout;
using std::endl;
using std::vector;
using std::set;
using std::string;
using std::stringstream;
using std::ostringstream;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using caffe::LayerParameter;
using caffe::NetParameter;

int g_layer_count;
int g_model_layers_count;

const int _string_size = 30;

const int _model_version = 1;

caffe::NetParameter g_proto;

caffe::NetParameter _model;

// shape of matrices
map<string, vector<int> > g_shape_map;

vector<LayerParameter> v_model_layers;

vector<LayerParameter> v_net_layers;

set<string>layer_types;

float *test_data;

// test data length
int input_data_count;

bool in_split;
map<string, int> split_pid_map;

//support ios mobile net, so need fold batch normal and scale to depthwise and pointwise
bool g_ios_gpu = false;
bool g_ios_gpu_classify = false;
enum Convolution_sub_type
{
    CONVOLUTION, DEPTHWISE, POINTWISE
};
Convolution_sub_type g_conv_sub_type = CONVOLUTION;

// macro to choose whether need quantification
#define NEED_QUANTI

/**
 replace all the input origin char with a new char
 */
void replace_all(std::string &str, char origin, char to){
    string::size_type str_length = str.length();
    for (int i = 0; i < str_length; ++i) {
        auto ch = str[i];
        if (ch == origin) {
            str.replace(i, 1, std::string(1, to));
        }
    }
}




bool get_data_from_file(string path, float *data) {
    char tag = ' ';
    ifstream model_file(path);
    if (!model_file) {
        throw string("cant't read the data file");
    }
    stringstream ss;
    ss << model_file.rdbuf();
    model_file.close();
    string inputString = ss.str();
    int length = inputString.length();
    int start = 0;
    vector<double> vector_data;
    for (int i = 0; i < length; i++) {
        if (inputString[i] == tag) {
            string sub = inputString.substr(start, i - start);
            vector_data.push_back(std::stod(sub));
            start = i + 1;
        }
    }
    if (vector_data.size() != input_data_count) {
        stringstream msg;
        msg << "input data length(" << vector_data.size()<<") != the size required by current network!";
        throw msg.str();
    }
    for (int i = 0; i < vector_data.size(); i++) {
        data[i] = vector_data[i];
    }
    return true;
}

/**
 * read from text proto
 * @param filepath
 * @param message
 * @return
 */
bool read_proto_from_text(const char *filepath, google::protobuf::Message *message) {
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open()) {

        throw string("proto text open failed! ");
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
}

/**
 * read from binary proto
 * @param filepath
 * @param message
 * @return
 */
bool read_proto_from_binary(const char *filepath, google::protobuf::Message *message) {
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open()) {
        throw string("proto binary open failed! ");
    }
    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

/**
 * get shape  from  BlobProto
 * @param proto
 * @return
 */
vector<int> get_shape(const caffe::BlobProto &proto) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
        // Using deprecated 4D Blob dimensions --
        // shape is (num, channels, height, width).
        shape.resize(4);
        shape[0] = proto.num();
        shape[1] = proto.channels();
        shape[2] = proto.height();
        shape[3] = proto.width();
    } else {
        shape.resize(proto.shape().dim_size());
        for (int i = 0; i < proto.shape().dim_size(); ++i) {
            shape[i] = proto.shape().dim(i);
        }
    }
    return shape;


}

/**
 * unfold shape vector to  string
 * @param shape
 * @return
 */
string get_shape_string(vector<int> shape) {
    ostringstream stream;

    for (int i = 0; i < shape.size(); ++i) {
        stream << shape[i];
        if (i < shape.size() - 1) {
            stream << ",";
        }
    }
    return stream.str();
}

/**
 * check map whether contains the key
 * @param shape_map
 * @param key
 * @return
 */
bool is_key_contain(map<string, vector<int> > shape_map, string key) {
    if (shape_map.find(key) != shape_map.end()) {
        return true;
    }
    return false;
}
/**
 * check set whether contains the key
 * @param shape_map
 * @param key
 * @return
 */
bool is_key_contain(set<string> types, string key) {
    if (types.find(key) != types.end()) {
        return true;
    }
    return false;
}

/**
 * for layers who don't change the input shape
 * @param shape_map
 * @param layer
 */
void copy_bottom_shape(map<string, vector<int> > &shape_map, const LayerParameter &layer) {
    if (layer.bottom_size() > 0) {
        string bottom_name = layer.bottom(0);
        for (int i = 0; i < layer.top_size(); ++i) {
            string top_name = layer.top(i);
            // ignore the <top_name> layer has been reshaped
            if (is_key_contain(shape_map, top_name)) {
                continue;

            }
            if (is_key_contain(shape_map, bottom_name)) {
                vector<int> bottom_shape = shape_map[bottom_name];

                shape_map.insert(make_pair(top_name, bottom_shape));
                cout << top_name << " shape = " << get_shape_string(shape_map[top_name]) << endl;

            } else {
                stringstream msg;
                msg<<"layer " << bottom_name << "'shape is not ready";
                throw msg.str();
            }
        }


    }

}

/**
 * clculate the layer shape
 * @param shape_map
 * @param layer
 */
void calcu_layer_shape(map<string, vector<int> > &shape_map, const LayerParameter &layer) {
    string type = layer.type();
    if (type == "Convolution") {
        if (layer.bottom_size() == 1) {
            string bottom_name = layer.bottom(0);
            string top_name = layer.top(0);
            if (is_key_contain(shape_map, top_name)) {
                return;

            }
            if (is_key_contain(shape_map, bottom_name)) {
                vector<int> bottom_shape = shape_map[bottom_name];
                const caffe::ConvolutionParameter &conv_param = layer.convolution_param();

                int num_output = conv_param.num_output();
                int kernel_size = conv_param.kernel_size(0);
                int pad = conv_param.pad_size() > 0 ? conv_param.pad(0) : 0;
                int stride = conv_param.stride_size() > 0 ? conv_param.stride(0) : 1;

                vector<int> top_shape;
                top_shape.push_back(bottom_shape[0]);
                top_shape.push_back(num_output);
                // width ==  height
                if (bottom_shape[2] == bottom_shape[3]) {
                    int n = (bottom_shape[2] - kernel_size + 2 * pad) / stride + 1;
                    top_shape.push_back(n);
                    top_shape.push_back(n);
                } else {
                    stringstream msg;
                    msg << bottom_name << "'s width != height !";
                    throw msg.str();
                }
                shape_map.insert(make_pair(top_name, top_shape));
                cout << top_name << " shape = " << get_shape_string(top_shape) << endl;


            } else {
                stringstream msg;
                msg << layer.name() << "bottom shape is not ready! ";
                throw msg.str();
            }

        } else {
            throw string("multiple inputs not supported yet!");
        }

    } else if (type == "Concat") {
        const caffe::ConcatParameter &concat_param = layer.concat_param();
        int concat_index = 0;
        if (concat_param.has_concat_dim()) {
            concat_index = concat_param.concat_dim();
        } else {
            concat_index = concat_param.axis();
        }
        string top_name = layer.top(0);
        vector<int> top_shape = shape_map[layer.bottom(0)];

        for (int i = 1; i < layer.bottom_size(); ++i) {
            top_shape[concat_index] += shape_map[layer.bottom(i)][concat_index];

        }
        shape_map.insert(make_pair(top_name, top_shape));
        cout << top_name << "shape =" << get_shape_string(top_shape) << endl;
    } else if (type == "InnerProduct") {
            string bottom_name = layer.bottom(0);
            string top_name = layer.top(0);
            if (is_key_contain(shape_map, top_name)) {
                return;
            }
            if (is_key_contain(shape_map, bottom_name)) {
                vector<int> bottom_shape = shape_map[bottom_name];
                const caffe::InnerProductParameter &innerProductParameter = layer.inner_product_param();

                int axis = innerProductParameter.axis();
                if (axis < 0) {
                    axis = bottom_shape.size() + axis;
                }
                int num_output = innerProductParameter.num_output();

                vector<int> top_shape = bottom_shape;
                top_shape.resize(axis + 1);
                // InnerProductLayer   change the channels
                top_shape[axis] = num_output;

                shape_map.insert(make_pair(top_name, top_shape));
                cout << top_name << " shape = " << get_shape_string(top_shape) << endl;

            } else {
                stringstream msg;
                msg << layer.name() << "bottom shape is not ready! ";
                throw msg.str();
            }

    } else if (type == "Pooling") {
        if (layer.bottom_size() == 1) {
            string bottom_name = layer.bottom(0);
            string top_name = layer.top(0);
            if (is_key_contain(shape_map, top_name)) {
                return;

            }
            if (is_key_contain(shape_map, bottom_name)) {
                vector<int> bottom_shape = shape_map[bottom_name];
                const caffe::PoolingParameter &pool_param = layer.pooling_param();
                int kernel_size = 0;
                int pad = 0;
                int stride = 1;
                if (pool_param.global_pooling()) {
                    kernel_size = bottom_shape[2];

                } else {
                    if (!pool_param.has_kernel_size()) {
                        stringstream msg;
                        msg<<layer.name()<<"'s pooling kernel_size should be configed!";
                        throw msg.str();
                    } else {
                        kernel_size = pool_param.kernel_size();
                    }
                    pad = pool_param.pad();
                    stride = pool_param.stride();

                }

                vector<int> top_shape;
                // Pooling layer just change the width & height
                top_shape.push_back(bottom_shape[0]);
                top_shape.push_back(bottom_shape[1]);
                // width ==  height
                if (bottom_shape[2] == bottom_shape[3]) {
                    int n = static_cast<int>(ceil(
                            static_cast<float>(bottom_shape[2] - kernel_size + 2 * pad) / stride)) + 1;
                    top_shape.push_back(n);
                    top_shape.push_back(n);
                } else {
                    stringstream msg;
                    msg << bottom_name << "width != height not supported yet";
                    throw msg.str();
                }
                shape_map.insert(make_pair(top_name, top_shape));
                cout << top_name << " shape = "  << get_shape_string(top_shape) << endl;

            } else {
                stringstream msg;
                msg << bottom_name << " shape is not ready";
                throw msg.str();
            }

        } else {
            throw string("multiple inputs not supported yet!");
        }

    } else {

        copy_bottom_shape(shape_map, layer);

    }
}

/**
 * calculate the blobs' shape & save in map
 * @param shape_map
 * @param layer
 */
void calcu_blobs_shape(map<string, vector<int> > &shape_map, const LayerParameter &layer) {
    
    if (g_ios_gpu && layer.type() == "BatchNorm"){
        if (layer.blobs_size()){
            vector<int> blob_shape = get_shape(layer.blobs(0));
            string blob_name = layer.name();
            blob_name.erase(blob_name.end() - 3, blob_name.end());
            stringstream s_blob_name;
            s_blob_name << blob_name << "_" << "1";
            shape_map.insert(make_pair(s_blob_name.str(), blob_shape));
            return;
        }
    }else if (g_ios_gpu && layer.type() == "Scale"){
        return;
    }
    
    for (int k = 0; k < layer.blobs_size(); ++k) {
        stringstream blob_name;
        blob_name << layer.name() << "_" << k;
        vector<int> blob_shape = get_shape(layer.blobs(k));
        
        if (g_ios_gpu && layer.type() == "Convolution" && layer.name().find("/dw") != string::npos){
            std::iter_swap(blob_shape.begin(), blob_shape.begin() + 1);
        }
        
        shape_map.insert(make_pair(blob_name.str(), blob_shape));
    }
}

/**
 * get the shape of input & calculate the data length for future use
 * @param shape_map
 * @param proto
 */
void read_input_shape(map<string, vector<int> > *shape_map, caffe::NetParameter proto) {
    vector<int> input_dimens;
    if (proto.input_dim_size() == 0) {
        auto input_layer = proto.layer(0);
        auto input_name = input_layer.top(0);

        auto input_param = input_layer.input_param();
        auto shape = input_param.shape(0);

        vector<int> shape_vec;
        input_data_count = 1;
        for (int i = 0; i < shape.dim_size(); ++i) {
            shape_vec.push_back(shape.dim(i));
            input_data_count *= shape.dim(i);
        }
        shape_map->insert(make_pair(input_name, shape_vec));
        return;
    }


    if (proto.input_dim_size() == 3) {
        input_dimens.push_back(1);

    }
    input_data_count = 1;

    for (int l = 0; l < proto.input_dim_size(); ++l) {
        input_dimens.push_back(proto.input_dim(l));
        input_data_count *= proto.input_dim(l);
    }
    shape_map->insert(make_pair(proto.input(0), input_dimens));
}

/**
 * get shapes string of layers
 * @param shape_map
 * @return
 */
string get_layer__shapes_string(map<string, vector<int> > *shape_map) {
    stringstream ss;
    for (auto it = shape_map->begin(); it != shape_map->end(); ++it) {
        string matrix_name = it->first;
        replace_all(matrix_name, '/', '_');
        ss << "\"" << matrix_name << "\":";
        ss << "[" << endl;
        ss << get_shape_string(it->second) << endl;
        ss << "]," << endl;
    }
    //remove last dot
    std::size_t found = ss.str().rfind(",");
    if (found != std::string::npos) {
        return ss.str().substr(0, found);
    }
    return ss.str();
}

/**
 * get the hash code of string
 * @param str
 * @return
 */
unsigned get_string_hash(string str) {
    const unsigned A = 54059;
    const unsigned B = 76963;
    unsigned res = 37;
    for (int i = 0; i < str.size(); i += 10) {
        char ch = str[i];
        res = (res * A) ^ (ch * B);
    }
    return res;
}

/**
 * append the hash code of json string to stringstream
 * @param ss
 */
void sign_json(stringstream &ss) {
    unsigned sign = get_string_hash(ss.str());
    ss << " " << sign << "=";
}

/**
 * transform layer types from caffe to mdl
 * @param layer_type
 * @return
 */
string get_mdl_layer_type(string layer_type) {
    if (layer_type == "InnerProduct") {
        return "FCLayer";
    } else if (layer_type == "ReLU") {
        return "ReluLayer";
    } else if (layer_type == "LRN") {
        return "LrnLayer";
    } else if (layer_type == "Convolution" && g_ios_gpu) {
        if (g_conv_sub_type == DEPTHWISE) {
            return "DepthwiseConvolutionLayer";
        } else if (g_conv_sub_type == POINTWISE) {
            return "PointwiseConvolutionLayer";
        }
        
    }
    return layer_type + "Layer";
}

/**
 * transform  pooling type to mdl
 * @param pool
 * @return
 */
string get_mdl_pooling_type(int pool) {
    string type;
    switch (pool) {
        case 0:
            type = "max";
            break;
        case 1:
            type = "ave";

            break;
        case 2:
            type = "stochastic";
            break;
        default:
            type = "max";
            break;

    }
    return type;
}
/**
 * transform  eltwise operation type to mdl
 * @param pool
 * @return
 */
string get_mdl_eltwise_type(int op) {
    string type;
    switch (op) {
        case 0:
            type = "product";
            break;
        case 1:
            type = "sum";

            break;
        case 2:
            type = "max";
            break;
        default:
            type = "sum";
            break;

    }
    return type;
}


/**
 * dump the json file
 * @param filename
 */
void dump_json(string filename) {
    stringstream json_string_stream;

    json_string_stream << "{" << endl;
    json_string_stream << "\"layer\": [" << endl;

    int index = 0;
    for (auto layer:v_net_layers) {
        string layer_name = layer.name();
        replace_all(layer_name, '/', '_');

        index++;

        if (layer.type() == "Input" || layer.type() == "Dropout") {
            continue;
        }
        
        if (g_ios_gpu && layer.type() == "Convolution"){
            if (layer.name().find("/dw") != string::npos){
                g_conv_sub_type = DEPTHWISE;
            }else if (layer.name().find("/sep") != string::npos){
                g_conv_sub_type = POINTWISE;
            }else{
                g_conv_sub_type = CONVOLUTION;
            }
        }
        
        if (g_ios_gpu && (layer.type() == "BatchNorm" || layer.type() == "Scale")){
            continue;
        }

        int _net_index = 0;
        for (_net_index = 0; _net_index < _model.layer_size(); ++_net_index) {
            if (_model.layer(_net_index).name() == layer.name()) {
                break;

            }
        }

        json_string_stream << "{" << endl;
        json_string_stream << "\"name\":\"" << layer_name << "\"," << endl;
        json_string_stream << "\"input\":[" << endl;
        for (int bottom_id = 0; bottom_id < layer.bottom_size(); ++bottom_id) {
            string blob_name = layer.bottom(bottom_id);
            replace_all(blob_name, '/', '_');
            json_string_stream << "\"" << blob_name << "\"";
            if (bottom_id < layer.bottom_size() - 1) {
                json_string_stream << ",";
            }
        }
        json_string_stream << "]," << endl;

        json_string_stream << "\"output\":[" << endl;
        for (int top_id = 0; top_id < layer.top_size(); ++top_id) {
            string blob_name = layer.top(top_id);
            replace_all(blob_name, '/', '_');
            json_string_stream << "\"" << blob_name << "\"";
            if (top_id < layer.top_size() - 1) {
                json_string_stream << ",";
            }
        }
        json_string_stream << "]," << endl;
        json_string_stream << "\"weight\":[" << endl;

        // if not found in model, just pass
        if (_net_index != _model.layer_size()) {
            const LayerParameter &model_layer = _model.layer(_net_index);
            for (int k = 0; k < model_layer.blobs_size(); ++k) {
                string model_layer_name = model_layer.name();
                replace_all(model_layer_name, '/', '_');
                json_string_stream << "\"" << model_layer_name << "_" << k << "\"";
                if (k != model_layer.blobs_size() - 1) {
                    json_string_stream << "," << endl;
                }
            }
            
            if (index < g_layer_count && model_layer.blobs_size() == 1 && g_ios_gpu && layer.type() == "Convolution"){
                if (v_net_layers[index].type() == "BatchNorm") {
                    string model_name = model_layer.name();
                    replace_all(model_name, '/', '_');
                    json_string_stream << ", \"" << model_name<< "_1" << "\"";
                }
            }
        }
        
        json_string_stream << "]," << endl;
        const char *type = layer.type().c_str();
        json_string_stream << "\"type\":\"" << get_mdl_layer_type(layer.type()) << "\"";
        if (in_split && strcmp(type, "Concat") != 0 && is_key_contain(layer_types,"Concat")) {
            int pid = split_pid_map[layer.bottom(0)];
            split_pid_map[layer.top(0)] = pid;
            json_string_stream << "," << endl;
            json_string_stream << "\"pid\":" << pid;
        }


        if (strcmp(type, "Convolution") == 0) {
            const caffe::ConvolutionParameter &conv_param = layer.convolution_param();
            json_string_stream << "," << endl;
            json_string_stream << "\"param\":{" << endl;
            json_string_stream << "\"output_num\":" << conv_param.num_output() << "," << endl;
            json_string_stream << "\"kernel_size\":" << conv_param.kernel_size(0) << "," << endl;
            int pad = conv_param.pad_size() != 0 ? conv_param.pad(0) : 0;
            json_string_stream << "\"pad\":" << pad << "," << endl;
            int stride = conv_param.stride_size() != 0 ? conv_param.stride(0) : 1;
            json_string_stream << "\"stride\":" << stride << "," << endl;
            json_string_stream << "\"bias_term\":" << conv_param.bias_term() << "," << endl;
            json_string_stream << "\"group\":" << conv_param.group() << endl;
            json_string_stream << "}" << endl;
        } else if (strcmp(type, "Scale") == 0) {
            const caffe::ScaleParameter &scale_param = layer.scale_param();
            json_string_stream << "," << endl;
            json_string_stream << "\"param\":{" << endl;
            json_string_stream << "\"bias_term\":" << scale_param.bias_term() << endl;
            json_string_stream << "}" << endl;
        } else if (strcmp(type, "InnerProduct") == 0) {
            const caffe::InnerProductParameter &inner_param = layer.inner_product_param();
            json_string_stream << "," << endl;
            json_string_stream << "\"param\":{" << endl;
            json_string_stream << "\"output_num\":" << inner_param.num_output() << endl;
            json_string_stream << "}" << endl;
        } else if (strcmp(type, "Pooling") == 0) {
            const caffe::PoolingParameter &poolingParameter = layer.pooling_param();
            json_string_stream << "," << endl;
            json_string_stream << "\"param\":{" << endl;
            json_string_stream << "\"type\":" << "\"" << get_mdl_pooling_type(poolingParameter.pool()) << "\"" << ","
                               << endl;
            if (poolingParameter.global_pooling()) {
                json_string_stream << "\"global_pooling\":" << "true" << endl;
            } else {
                json_string_stream << "\"kernel_size\":" << poolingParameter.kernel_size() << "," << endl;
                json_string_stream << "\"pad\":" << poolingParameter.pad() << "," << endl;
                json_string_stream << "\"stride\":" << poolingParameter.stride() << endl;

            }
            json_string_stream << "}" << endl;
        } else if (strcmp(type, "Eltwise") == 0) {
            const caffe::EltwiseParameter &eltwiseParameter = layer.eltwise_param();
            json_string_stream << "," <<endl;
            json_string_stream << "\"param\":{" << endl;
            json_string_stream << "\"type\":" << "\"" << get_mdl_eltwise_type(eltwiseParameter.operation()) << "\"" ;
            if (strcmp(type, "sum") == 0 && eltwiseParameter.coeff_size()) {
                json_string_stream << "," << endl;
                json_string_stream << "\"coeffs\":[";
                for (int i = 0; i < eltwiseParameter.coeff_size(); ++i) {
                    json_string_stream << eltwiseParameter.coeff(i);
                    if (i < eltwiseParameter.coeff_size() - 1) {
                        json_string_stream << ",";

                    }

                }
                json_string_stream << "]" <<endl;

            }
            json_string_stream << "}" << endl;

        } else if (strcmp(type, "LRN") == 0) {
            const caffe::LRNParameter &lrn_param = layer.lrn_param();
            json_string_stream << "," << endl;
            json_string_stream << "\"param\":{" << endl;
            json_string_stream << "\"local_size\":" << lrn_param.local_size() << "," << endl;
            json_string_stream << "\"alpha\":" << lrn_param.alpha() << "," << endl;
            json_string_stream << "\"beta\":" << lrn_param.beta() << endl;
            json_string_stream << "}" << endl;

        } else if (strcmp(type, "Split") == 0) {
            in_split = true;
            for (int i = 0; i < layer.top_size(); ++i) {
                split_pid_map[layer.top(i)] = i + 1;
            }
        } else if (strcmp(type, "Concat") == 0) {
            in_split = false;
        }
        if (index == g_layer_count) {
            
            //if ios gpu classify add softmax layer
            if (g_ios_gpu_classify && layer.type() != "Softmax") {
                json_string_stream << "}," << endl;
                json_string_stream << "{" << endl;
                json_string_stream << "\"name\":\"" << "Softmax" << "\"," << endl;
                json_string_stream << "\"input\":[" << endl;
                for (int top_id = 0; top_id < layer.top_size(); ++top_id) {
                    string blob_name = layer.top(top_id);
                    replace_all(blob_name, '/', '_');
                    json_string_stream << "\"" << blob_name << "\"";
                    if (top_id < layer.top_size() - 1) {
                        json_string_stream << ",";
                    }
                }
                
                json_string_stream << "]," << endl;
                json_string_stream << "\"type\" : " << "\"SoftmaxLayer\"," <<endl;
                json_string_stream << "\"output\":[" << endl;
                json_string_stream << "\"Softmax\"";
                json_string_stream << "]" << endl;
            }
            
            json_string_stream << "}" << endl;
        } else {
            json_string_stream << "}," << endl;
        }
    }

    // dump matrixs
    json_string_stream << "]," << endl;
    json_string_stream << "\"matrix\":{" << endl;

    // calculate the top shape for current layer & save in map
    for (auto layer:v_net_layers) {
        calcu_layer_shape(g_shape_map, layer);

    }
    // calculate the blobs(weights & bias) shape for current layer & save in map
    for (int l = 0; l < g_model_layers_count; ++l) {
        const LayerParameter &model_layer = _model.layer(l);
        calcu_blobs_shape(g_shape_map, model_layer);
    }
    json_string_stream << get_layer__shapes_string(&g_shape_map) << endl;
    
    if (g_ios_gpu_classify && v_net_layers.back().type() != "Softmax"){
        json_string_stream << "," << "\"Softmax\" : ";
        json_string_stream << "[";
        json_string_stream << get_shape_string(g_shape_map[v_net_layers.back().name()]);
        json_string_stream << "]";
    }
    
    json_string_stream << "}," << endl;
    json_string_stream << " \"meta\": {" << endl;
    json_string_stream << "\"model_version\":" << _model_version << endl;
    json_string_stream << "}" << endl;
    json_string_stream << "}";
    sign_json(json_string_stream);
    std::ofstream json_file(filename);
    if (!json_file.is_open()) {
        throw string("jsonFile open fail");
    }
    json_file << json_string_stream.str();

    json_file.close();
    cout << "finish dump json!" << endl;

}

/**
 * transpose matrix in advance
 * @param data
 * @param shape
 * @return
 */
float *trans_matrix(const float *data, vector<int> shape) {

    int m = shape[0];
    int n = shape[1];

    float *trans = new float[m * n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {

            trans[i * m + j] = data[j * n + i];
        }
    }
    return trans;

}

/**
 * copy matrix
 * @param src
 * @param dest
 * @param length
 */
void copy_matrix(const float *src, float *dest, int length) {
    for (int i = 0; i < length; ++i) {
        dest[i] = src[i];

    }
}

void dump_without_quantification(string filename) {
    int total_size = 0;
    int matrix_count = 0;

    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            matrix_count++;
            const caffe::BlobProto &blob = model_layer.blobs(i);
            total_size += blob.data_size() * sizeof(float);
        }
    }
    if (test_data) {
        matrix_count++;
        total_size += input_data_count * sizeof(float);
    }

    total_size += matrix_count * (_string_size * sizeof(char) + sizeof(int));
    total_size += 3 * sizeof(int);

    FILE *out_file = fopen(filename.c_str(), "wb");
    fwrite(&total_size, sizeof(int), 1, out_file);
    fwrite(&_model_version, sizeof(int), 1, out_file);
    fwrite(&matrix_count, sizeof(int), 1, out_file);
    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            const caffe::BlobProto &blob = model_layer.blobs(i);
            int data_size = blob.data_size();
            fwrite(&data_size, sizeof(int), 1, out_file);
        }
    }
    if (test_data) {
        fwrite(&input_data_count, sizeof(int), 1, out_file);
    }
    char matrix_name[_string_size];
    int matrix_name_count = 0;
    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            stringstream blob_name;
            blob_name << model_layer.name() << "_" << i;
            strcpy(matrix_name, blob_name.str().c_str());
            matrix_name_count++;
            fwrite(matrix_name, sizeof(char), _string_size, out_file);
        }
    }

    if (test_data) {
        string matrix_name_data = "data";
        strcpy(matrix_name, matrix_name_data.c_str());
        fwrite(matrix_name, sizeof(char), _string_size, out_file);

    }

    int matrix_index = 0;
    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {

            const caffe::BlobProto &blob = model_layer.blobs(i);
            float *tmp = new float[blob.data_size()];
            copy_matrix(blob.data().data(), tmp, blob.data_size());
            if (strcmp(model_layer.type().c_str(), "InnerProduct") == 0 && 0 == i) {
                bool transpose = model_layer.inner_product_param().transpose();
                if (!transpose) {
                    stringstream blob_name;
                    blob_name << model_layer.name() << "_" << i;
                    tmp = trans_matrix(blob.data().data(), g_shape_map[blob_name.str()]);
                }

            }
            for (int j = 0; j < blob.data_size(); j++) {
                float value = tmp[j];
                fwrite(&value, sizeof(float), 1, out_file);
            }

            matrix_index++;
        }
    }
    if (test_data) {
        for (int i = 0; i < input_data_count; i++) {
            float value = test_data[i];
            fwrite(&value, sizeof(float), 1, out_file);
        }
        matrix_index++;
    }
    fclose(out_file);
    cout << "finish dump the binary file without quantification for the model!" << endl;

}

/**
 * dump_with_quantification
 * @param filename
 */
void dump_with_quantification(string filename) {
    int total_size = 0;
    int matrix_count = 0;

    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            matrix_count++;
            const caffe::BlobProto &blob = model_layer.blobs(i);
            total_size += blob.data_size() * sizeof(uint8_t);
        }
    }

    if (test_data) {
        matrix_count++;
        total_size += input_data_count * sizeof(uint8_t);
    }

    total_size += matrix_count * (_string_size * sizeof(char) + sizeof(int) + 2 * sizeof(float));
    total_size += 3 * sizeof(int);

    FILE *out_file = fopen(filename.c_str(), "wb");
    fwrite(&total_size, sizeof(int), 1, out_file);
    fwrite(&_model_version, sizeof(int), 1, out_file);
    fwrite(&matrix_count, sizeof(int), 1, out_file);

    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            const caffe::BlobProto &blob = model_layer.blobs(i);
            int data_size = blob.data_size();
            fwrite(&data_size, sizeof(int), 1, out_file);
        }
    }
    if (test_data) {
        fwrite(&input_data_count, sizeof(int), 1, out_file);
    }

    vector<float> min_values;
    vector<float> max_values;

    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            const caffe::BlobProto &blob = model_layer.blobs(i);

            float min_value = std::numeric_limits<float>::max();
            float max_value = std::numeric_limits<float>::min();
            for (int j = 0; j < blob.data_size(); ++j) {
                min_value = min(min_value, blob.data().data()[j]);
                max_value = max(max_value, blob.data().data()[j]);
            }
            min_values.push_back(min_value);
            max_values.push_back(max_value);
            fwrite(&min_value, sizeof(float), 1, out_file);
            fwrite(&max_value, sizeof(float), 1, out_file);
        }
    }

    if (test_data) {
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::min();
        for (int i = 0; i < input_data_count; i++) {
            min_value = min(min_value, test_data[i]);
            max_value = max(max_value, test_data[i]);
        }
        min_values.push_back(min_value);
        max_values.push_back(max_value);
        fwrite(&min_value, sizeof(float), 1, out_file);
        fwrite(&max_value, sizeof(float), 1, out_file);
    }
    char matrix_name[_string_size];
    int matrix_name_count = 0;
    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {
            stringstream blob_name;
            blob_name << model_layer.name() << "_" << i;
            strcpy(matrix_name, blob_name.str().c_str());
            matrix_name_count++;
            string matrix_name_str = string(matrix_name);
            replace_all(matrix_name_str, '/', '_');
            fwrite(matrix_name_str.c_str(), sizeof(char), _string_size, out_file);
        }
    }

    if (test_data) {
        string matrix_name_data = "data";
        strcpy(matrix_name, matrix_name_data.c_str());
        fwrite(matrix_name, sizeof(char), _string_size, out_file);

    }

    int matrix_index = 0;
    for (auto model_layer:v_model_layers) {
        for (int i = 0; i < model_layer.blobs_size(); ++i) {

            const caffe::BlobProto &blob = model_layer.blobs(i);
            float min_value = min_values[matrix_index];
            float max_value = max_values[matrix_index];
            float *tmp = new float[blob.data_size()];
            copy_matrix(blob.data().data(), tmp, blob.data_size());
            if (strcmp(model_layer.type().c_str(), "InnerProduct") == 0 && 0 == i) {
                bool transpose = model_layer.inner_product_param().transpose();
                if (!transpose) {
                    stringstream blob_name;
                    blob_name << model_layer.name() << "_" << i;
                    tmp = trans_matrix(blob.data().data(), g_shape_map[blob_name.str()]);
                }

            }
            for (int j = 0; j < blob.data_size(); j++) {
                float value = tmp[j];
                uint8_t factor = (uint8_t) round((value - min_value) / (max_value - min_value) * 255);
                fwrite(&factor, sizeof(uint8_t), 1, out_file);
            }

            matrix_index++;
        }
    }
    if (test_data) {
        float min_value = min_values[matrix_index];
        float max_value = max_values[matrix_index];
        for (int i = 0; i < input_data_count; i++) {
            float value = test_data[i];
            uint8_t factor = (uint8_t) round((value - min_value) / (max_value - min_value) * 255);
            fwrite(&factor, sizeof(uint8_t), 1, out_file);
        }
        matrix_index++;
    }
    fclose(out_file);
    cout << "finish dump the binary file for the model!" << endl;
}

/**
 * get split layer name
 * @param layer_name
 * @param blob_name
 * @param blob_idx
 * @return
 */
string get_split_layer_name(const string &layer_name, const string &blob_name,
                            const int blob_idx) {
    ostringstream split_layer_name;
    split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
                     << "_split";
    return split_layer_name.str();
}

/**
 * get split blob name
 * @param layer_name
 * @param blob_name
 * @param blob_idx
 * @param split_idx
 * @return
 */
string get_split_blob_name(const string &layer_name, const string &blob_name,
                           const int blob_idx, const int split_idx) {
    ostringstream split_blob_name;
    split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
                    << "_split_" << split_idx;
    return split_blob_name.str();
}

/**
 * configure splitLayer
 * @param layer_name
 * @param blob_name
 * @param blob_idx
 * @param split_count
 * @param split_layer_param
 */
void config_splitLayer(const string &layer_name, const string &blob_name,
                       const int blob_idx, const int split_count,
                       LayerParameter *split_layer_param) {
    split_layer_param->Clear();
    split_layer_param->add_bottom(blob_name);
    split_layer_param->set_name(get_split_layer_name(layer_name, blob_name, blob_idx));
    split_layer_param->set_type("Split");
    for (int k = 0; k < split_count; ++k) {
        split_layer_param->add_top(
                get_split_blob_name(layer_name, blob_name, blob_idx, k));
    }
}


void setup_splits(NetParameter &param, NetParameter *param_split) {
    param_split->CopyFrom(param);
    param_split->clear_layer();
    map<string, pair<int, int> > blobs_to_layer_tops;
    map<pair<int, int>, pair<int, int> > bottom_ids_to_top_ids;
    map<pair<int, int>, int> tops_as_bottom_count;
    map<pair<int, int>, int> tops_split_index;
    map<int, string> index_to_layer_name;

    for (int i = 0; i < param.layer_size(); ++i) {
        const LayerParameter &layer_param = param.layer(i);
        index_to_layer_name[i] = layer_param.name();

        for (int j = 0; j < layer_param.bottom_size(); ++j) {
            const string &blob_name = layer_param.bottom(j);
            if (blobs_to_layer_tops.find(blob_name) ==
                blobs_to_layer_tops.end()) {
                stringstream msg;
                msg << "Unknown bottom blob '" << blob_name << "' (layer '"
                    << layer_param.name() << "', bottom index " << j << ")";
                if (blob_name == "data") {
                    cout<< "please make sure the input is described as a layer in caffe.prototxt"<<endl;
                }

                throw msg.str();


            }
            const pair<int, int> &bottom_idx = make_pair(i, j);
            const pair<int, int> &top_idx = blobs_to_layer_tops[blob_name];
            bottom_ids_to_top_ids[bottom_idx] = top_idx;
            ++tops_as_bottom_count[top_idx];
        }
        for (int j = 0; j < layer_param.top_size(); ++j) {
            const string &blob_name = layer_param.top(j);
            blobs_to_layer_tops[blob_name] = make_pair(i, j);
        }

    }
    for (int i = 0; i < param.layer_size(); ++i) {
        LayerParameter *layer_param = param_split->add_layer();
        layer_param->CopyFrom(param.layer(i));
        for (int j = 0; j < layer_param->bottom_size(); ++j) {
            const pair<int, int> &top_idx =
                    bottom_ids_to_top_ids[make_pair(i, j)];
            const int split_count = tops_as_bottom_count[top_idx];
            if (split_count > 1) {
                const string &layer_name = index_to_layer_name[top_idx.first];
                const string &blob_name = layer_param->bottom(j);
                layer_param->set_bottom(j, get_split_blob_name(layer_name,
                                                               blob_name, top_idx.second,
                                                               tops_split_index[top_idx]++));
            }
        }
        // Create split layer for any top blobs used by other layer as bottom
        // blobs more than once.
        for (int j = 0; j < layer_param->top_size(); ++j) {
            const pair<int, int> &top_idx = make_pair(i, j);
            const int split_count = tops_as_bottom_count[top_idx];
            if (split_count > 1) {
                const string &layer_name = index_to_layer_name[i];
                const string &blob_name = layer_param->top(j);
                // add split layer
                LayerParameter *split_layer_param = param_split->add_layer();
                config_splitLayer(layer_name, blob_name, j, split_count, split_layer_param);

            }
        }
    }


}

int main(int argc, char **args) {
    try {
        
        const char *ios_gpu_input = args[argc - 1];
        if (string(ios_gpu_input) == "-ios_gpu_mobilenet"){
            g_ios_gpu = true;
        }
        if (string(ios_gpu_input) == "-ios_gpu_mobilenet_classify"){
            g_ios_gpu = true;
            g_ios_gpu_classify =true;
        }
        
        const char *_caffe_proto = args[1];
        const char *_caffe_model = args[2];

        const char *mdl_json = "model.min.json";
#ifdef NEED_QUANTI
        const char *mdl_data = "data.min.bin";
#else
        const char *mdl_data = "data.bin";
#endif
        //g_proto  -- caffe::NetParameter 类型
        bool _success1 = read_proto_from_text(_caffe_proto, &g_proto);
        if (!_success1) {
            throw string("read_proto_from_text failed");
        }
        bool _success2 = read_proto_from_binary(_caffe_model, &_model);
        if (!_success2) {
            throw string("read proto from binary failed");
        }
        NetParameter para;
        if (g_ios_gpu){
            para = g_proto;
        }else{
            setup_splits(g_proto, &para);
        }        // initialize input shape
        read_input_shape(&g_shape_map, para);
        // read test data
        if (argc >= 4) {
            const char *_test_data = args[3];
            test_data = new float[input_data_count];
            bool _success3 = get_data_from_file(_test_data, test_data);
        }
        g_layer_count = para.layer_size();
        g_model_layers_count = _model.layer_size();
        for (int i = 0; i < g_layer_count; ++i) {
            const LayerParameter &layer = para.layer(i);
            v_net_layers.push_back(layer);
            layer_types.insert(layer.type());

        }
        for (string type:layer_types) {
            cout << type<< endl;
        }

        for (int j = 0; j < g_model_layers_count; j++) {
            const LayerParameter &model_layer = _model.layer(j);
            v_model_layers.push_back(model_layer);
        }
        dump_json(mdl_json);
#ifdef NEED_QUANTI
        dump_with_quantification(mdl_data);
#else
        dump_without_quantification(mdl_data);
#endif

    } catch (const string &msg) {
        cout << msg << endl;
    }

}


