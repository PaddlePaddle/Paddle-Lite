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
#include "loader/loader.h"

#include <fstream>
#include <fcntl.h>
#include <zconf.h>
#include <errno.h>
#include <sys/mman.h>

#include "base/matrix.h"

using std::ifstream;
using std::ofstream;

namespace mdl {
    Loader *Loader::_instance = nullptr;

    Loader *Loader::shared_instance() {
        if (_instance == nullptr) {
            _instance = new Loader();
        }
        return _instance;
    }

    char *get_binary_data(string filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        if (file == nullptr) {
            throw_exception("can't open binary file");
        }

        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        if (size <= 0 || size < sizeof(int)) {
            throw_exception("binary file size is too small");
        }
        rewind(file);

        char *data = new char[size];
        size_t bytes_read = fread(data, 1, size, file);
        if (bytes_read != size) {
            throw_exception("read binary file bytes do not match with fseek");
        }
        fclose(file);

        int size_in_header = *(int *) data;
        if (size_in_header != size) {
            throw_exception("binary file size do not match with header meta info");
        }

        return data;
    }

    bool Loader::load(string model_path, string weights_path) {
        _loaded = false;
        if (!_cleared) {
            throw_exception("Loader must be cleared before reloading");
        }
        _cleared = false;
        bool success = load_with_quantification(model_path, weights_path);
        _loaded = success;
        return success;
    }

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
     * get model string with signature validation
     *
     * @param model_path
     * @return
     */
    string get_model_string_from_file(string model_path) {
        ifstream model_file(model_path);
        if (!model_file) {
            throw_exception("can't open model json file");
        }
        stringstream ss;
        ss << model_file.rdbuf();
        model_file.close();
        string model_file_string = ss.str();
        if (model_file_string.size() <= 0) {
            throw_exception("model file string is empty");
        }
        size_t end_pos = model_file_string.rfind("=");
        if (end_pos == string::npos) {
            throw_exception("can't find end symbol in model file");
        }
        size_t blank_pos = model_file_string.rfind(" ");
        if (blank_pos == string::npos) {
            throw_exception("can't find blank splitter in model file");
        }
        if (blank_pos >= end_pos) {
            throw_exception("blank splitter is behind end symbol in model file");
        }
        string model_string = model_file_string.substr(0, blank_pos);

        // get sign_string from the end of the json file
        string sign_string = model_file_string.substr(blank_pos + 1, end_pos - blank_pos - 1);
        if (model_string.size() <= 0) {
            throw_exception("model string is empty");
        }
        if (sign_string.size() <= 0) {
            throw_exception("model sign string is empty");
        }
        // get the expected_sign_string from model_string
        size_t sign = get_string_hash(model_string);
        ss.str("");
        ss << sign;
        string sign_string_expected = ss.str();
        if (sign_string != sign_string_expected) {
            throw_exception("model sign is not expected");
        }
        return model_string;
    }

    /**
     *
     * @param model_path
     * @param weights_path
     * @return
     */
    /*
     * version: version code
     * model_count: num of layers
     * sizei:num of params for the ith layer
     * min max: the min & max value in the params of each layer
     *
     * ||  int  ||    int     || int   || int  || ...  ||float(2) || float(2) || ... || char(30)   || char(30)  || ... || data
     * ||version|| model_count|| size1 ||size2 || ...  || min max ||  min max || ... || layer_name || layer_name|| ... || data
     */
    bool Loader::load_with_quantification(string model_path, string weights_path) {
        string model_string = get_model_string_from_file(model_path);
        string error;
        _model = Json::parse(model_string, error);
        if (error.size() > 0) {
            throw_exception("json parser error %s", error.c_str());
        }

        for (auto &pair: _model["matrix"].object_items()) {
            string name = pair.first;
            Json config = pair.second;
            Matrix *matrix = new Matrix(config);
            matrix->reallocate();
            if (matrix == nullptr) {
                throw_exception("can't create %s matrix", name.c_str());
            }
            if (name == matrix_name_data) {
                _matrices[matrix_name_test_data] = matrix;
                Matrix *data_matrix = new Matrix(config);
                if (data_matrix == nullptr) {
                    throw_exception("can't create data matrix");
                }
                _matrices[matrix_name_data] = data_matrix;
            } else {
                _matrices[name] = matrix;
            }
        }
        // validate the version code in the json file
        int model_version_from_json = _model["meta"]["model_version"].int_value();
        if (model_version != model_version_from_json) {
            throw_exception("model version from cpp does not match with version from json config");
        }
        char *original_binary_data = get_binary_data(weights_path);
        char *binary_data = original_binary_data;
        if (binary_data == nullptr) {
            throw_exception("can't load binary file");
        }

        try {
            if (model_version != *((int *) binary_data + 1)) {
                throw_exception("model version from cpp does not match with version from binary data");
            }
            int model_count = *((int *) binary_data + 2);
            if (model_count <= 0) {
                throw_exception("model count is too small");
            }
            binary_data = binary_data + 3 * sizeof(int);

            // the num of params of each layer
            vector<int> model_sizes(model_count);
            for (int i = 0; i < model_count; i++) {
                model_sizes[i] = *((int *) binary_data + i);
            }
            binary_data = binary_data + model_count * sizeof(int);
            vector<float> model_mins(model_count);
            vector<float> model_maxs(model_count);
            for (int i = 0; i < model_count; i++) {
                model_mins[i] = *((float *) binary_data + i * 2);
                model_maxs[i] = *((float *) binary_data + i * 2 + 1);
            }
            binary_data = binary_data + model_count * 2 * sizeof(float);
            vector<string> model_names(model_count);
            for (int i = 0; i < model_count; i++) {
                model_names[i] = string((char *) binary_data + i * string_size * sizeof(char));
            }
            binary_data = binary_data + model_count * string_size * sizeof(char);

            uint8_t *uint8_data = (uint8_t *) binary_data;
            for (int i = 0; i < model_count; i++) {
                int model_size = model_sizes[i];
                string model_name = model_names[i];
                if (model_name == matrix_name_data) {
                    model_name = matrix_name_test_data;
                }
                if (_matrices.find(model_name) == _matrices.end()) {
                    throw_exception("can't find %s in matrices when load binary file", model_name.c_str());
                }
                Matrix *matrix = _matrices[model_name];
                if (matrix->count() != model_size) {
                    throw_exception("matrix count does not match between json and binary file");
                }
                // inverse quantification
                const float min_value = model_mins[i];
                const float max_value = model_maxs[i];
                const float factor = (max_value - min_value) / 255.0;
                float *matrix_data = matrix->get_data();
                for (int j = 0; j < model_size / 4; j++) {
                    matrix_data[0] = uint8_data[0] * factor + min_value;
                    matrix_data[1] = uint8_data[1] * factor + min_value;
                    matrix_data[2] = uint8_data[2] * factor + min_value;
                    matrix_data[3] = uint8_data[3] * factor + min_value;
                    matrix_data += 4;
                    uint8_data += 4;
                }
                for (int j = 0; j < model_size % 4; j++) {
                    matrix_data[j] = uint8_data[j] * factor + min_value;
                }
                uint8_data += (model_size % 4);
            }
        } catch (...) {
            if (original_binary_data != nullptr) {
                delete original_binary_data;
                original_binary_data = nullptr;
            }
            throw;
            return false;
        }

        if (original_binary_data != nullptr) {
            delete original_binary_data;
            original_binary_data = nullptr;
        }
        return true;
    }

#ifdef NEED_DUMP
    /**
     * non quantification load method
     * @param model_path
     * @param weights_path
     * @return
     */
    bool Loader::load_without_quantification(string model_path, string weights_path) {
        string model_string = get_model_string_from_file(model_path);
        string error;
        _model = Json::parse(model_string, error);
        if (error.size() > 0) {
            throw_exception("json parser error %s", error.c_str());
        }

        for (auto &pair: _model["matrix"].object_items()) {
            string name = pair.first;
            Json config = pair.second;
            Matrix *matrix = new Matrix(config);
            matrix->reallocate();
            if (name == matrix_name_data) {
                _matrices[matrix_name_test_data] = matrix;
                _matrices[matrix_name_data] = new Matrix(config);
            } else {
                _matrices[name] = matrix;
            }
        }

        int model_version_from_json = _model["meta"]["model_version"].int_value();
        if (model_version != model_version_from_json) {
            return false;
        }
        char *original_binary_data = get_binary_data(weights_path);
        char *binary_data = original_binary_data;
        if (binary_data == nullptr) {
            return false;
        }

        if (model_version != *((int *)binary_data + 1)) {
            return false;
        }
        int model_count = *((int *)binary_data + 2);
        if (model_count <= 0) {
            return false;
        }

        binary_data = binary_data + 3 * sizeof(int);
        vector<int> model_sizes(model_count);
        for (int i = 0; i < model_count; i++) {
            model_sizes[i] = *((int *)binary_data + i);
        }
        binary_data = binary_data + model_count * sizeof(int);
        vector<string> model_names(model_count);
        for (int i = 0; i < model_count; i++) {
            model_names[i] = string((char *)binary_data + i * string_size * sizeof(char));
        }
        binary_data = binary_data + model_count * string_size * sizeof(char);
        float *float_data = (float *)binary_data;
        for (int i = 0; i < model_count; i++) {
            int model_size = model_sizes[i];
            string model_name = model_names[i];
            if (model_name == matrix_name_data) {
                model_name = matrix_name_test_data;
            }
            if (_matrices.find(model_name) == _matrices.end()) {
                return false;
            }
            Matrix *matrix = _matrices[model_name];
            std::copy(float_data, float_data + model_size, matrix->get_data());
            float_data += model_size;
        }

        delete original_binary_data;
        return true;
    }
#endif

    void Loader::clear() {
        _loaded = false;
        for (auto &pair: _matrices) {
            Matrix *matrix = pair.second;
            delete matrix;
        }
        _matrices.clear();
        _cleared = true;
    }
};
