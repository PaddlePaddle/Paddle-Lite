

#include "src/enforce.h"
#include "src/var_desc.h"
#include "src/program_desc.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include "src/framework.pb-c.h"
#include "src/protobuf-c.h"
#include <fstream>
#include <iostream>
#include <limits>

const size_t kSize64 = sizeof(uint64_t);
const size_t kSize32 = sizeof(uint32_t);
const int minimal_fold_size = 2;
float max_entropy = 0.0;

float entropy(std::vector<uint8_t> &factors) {
    int n = factors.size();
    std::vector<int> counts(256);
    for (uint8_t &factor : factors) {
        counts[factor]++;
    }
    float res = 1.0;
    float shift = 100000.0;
    for (int i = 0; i < 256; i++) {
        res *= (counts[i] + shift) / (n + shift);
    }
    return 1.0 / res;
}

char *Get_binary_data(const std::string &filename) {

    FILE *file = fopen(filename.c_str(), "rb");

    PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                          filename.c_str());
    fseek(file, 0, SEEK_END);
    int64_t size = ftell(file);

    PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
    rewind(file);
    auto *data = new char[size];
    size_t bytes_read = fread(data, 1, static_cast<size_t>(size), file);
    PADDLE_MOBILE_ENFORCE(bytes_read == size,
                          "read binary file bytes do not match with fseek");
    fclose(file);
    return data;
}


static size_t ReadBuffer(const char *file_name, uint8_t **out) {
    FILE *fp;
    fp = fopen(file_name, "rb");
    PADDLE_MOBILE_ENFORCE(fp != nullptr, " %s open failed !", file_name);
    fseek(fp, 0, SEEK_END);
    auto size = static_cast<size_t>(ftell(fp));
    rewind(fp);
    *out = reinterpret_cast<uint8_t *>(malloc(size));
    size_t cur_len = 0;
    size_t nread;
    while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
        cur_len += nread;
    }
    fclose(fp);
    return cur_len;
}

std::shared_ptr<ProgramDesc> loadParams(const std::string &model_path) {
    PaddleMobile__Framework__Proto__ProgramDesc *c_program;
    uint8_t *buf = nullptr;
    size_t read_size = ReadBuffer(model_path.c_str(), &buf);
    PADDLE_MOBILE_ENFORCE(buf != nullptr, "read from __model__ is null");
    c_program = paddle_mobile__framework__proto__program_desc__unpack(
            nullptr, read_size, buf);
    PADDLE_MOBILE_ENFORCE(c_program != nullptr, "program is null");
    auto originProgramDesc = std::make_shared<ProgramDesc>(c_program);
    return originProgramDesc;

}

void LoadWithDumpForInt8(const paddle_mobile::framework::VarDesc &var_desc, char **dataP, FILE *out_file, int quantification_fold) {
    // 1. version
    uint32_t version = *reinterpret_cast<uint32_t *>(*dataP);

    // write version
    fwrite(&version, kSize32, 1, out_file);

    *dataP += kSize32;

    // 2 Lod information
    auto *lod_level_ptr = new uint64_t();
    memcpy(lod_level_ptr, *dataP, kSize64);

    uint64_t lod_level = 0;
    // write lod Information
    fwrite(&lod_level, kSize64, 1, out_file);
    delete lod_level_ptr;

    *dataP += kSize64;

    for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size = *reinterpret_cast<uint64_t *>(*dataP);
        // write lod size
        fwrite(&size, kSize64, 1, out_file);
        (*dataP) += kSize64;

        std::vector<size_t> tmp(size / sizeof(size_t));
        for (unsigned long &k : tmp) {
            k = *reinterpret_cast<size_t *>(*dataP);
            (*dataP) += sizeof(size_t);
        }
        // write lod size vector
        fwrite(&tmp, sizeof(size_t), tmp.size(), out_file);
    }

    // 3. tensor version
    uint32_t tensor_version = *reinterpret_cast<uint32_t *>(*dataP);
    // write tensor version
    fwrite(&tensor_version, kSize32, 1, out_file);
    (*dataP) += kSize32;

    // 4. tensor desc
    int32_t size = *reinterpret_cast<int32_t *>(*dataP);
    // write tensor desc
    fwrite(&size, sizeof(int32_t), 1, out_file);
    (*dataP) += sizeof(int32_t);

    std::unique_ptr<char[]> buf(new char[size]);
    for (int m = 0; m < size; ++m) {
        buf.get()[m] = (*dataP)[m];
    }

    fwrite(buf.get(), sizeof(char), static_cast<size_t>(size), out_file);
    (*dataP) += (sizeof(char) * size);

    const paddle_mobile::framework::TensorDesc &desc = var_desc.Tensor_desc();
    int memory_size = 1;
    for (auto l : desc.Dims()) {
        memory_size *= l;
    }

    void *memory = nullptr;
    int type_size = 0;
    switch (desc.DataType()) {
        case paddle_mobile::framework::VARTYPE_TYPE_FP16:
            type_size = 2;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_FP32:
            type_size = 4;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_FP64:
            type_size = 8;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_INT32:
            type_size = 4;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_INT64:
            type_size = 8;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_BOOL:
            type_size = 1;
            break;
        default:
            break;
    }
    size_t tensorSize = sizeof(char) * memory_size * type_size;

    memory = new char[tensorSize];

    for (int n = 0; n < tensorSize; ++n) {
        static_cast<char *>(memory)[n] = (*dataP)[n];
    }
    *dataP += tensorSize;

    quantification_fold = std::min(std::max(1, memory_size / minimal_fold_size), quantification_fold);
    int step = std::max(memory_size / quantification_fold, 1);

    int visited_fold = 0;
    while (visited_fold * step < memory_size) {
        // for float 32
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::min();

        for (int k = visited_fold * step; k < std::min((visited_fold + 1) * step, memory_size); ++k) {
            min_value = std::min(min_value, static_cast<float *> (memory)[k]);
            max_value = std::max(max_value, static_cast<float *> (memory)[k]);
        }

        fwrite(&min_value, sizeof(float), 1, out_file);
        fwrite(&max_value, sizeof(float), 1, out_file);

        std::vector<uint8_t> factors;
        for (int g = visited_fold * step; g < std::min((visited_fold + 1) * step, memory_size); ++g) {
            float value = static_cast<float *> (memory)[g];
            auto factor = (uint8_t) round((value - min_value) / (max_value - min_value) * 255);
            factors.push_back(factor);
            fwrite(&factor, sizeof(uint8_t), 1, out_file);
        }
        max_entropy = fmax(max_entropy, entropy(factors));
        visited_fold++;
    }
}

void
quantificate_combined_int8(const std::string &model_path, const std::string &param_path, const std::string &param_min_path, int quantification_fold) {
    auto program = loadParams(model_path);
    char *origin_data = Get_binary_data(param_path);
    char *data = origin_data;
    FILE *out_file = fopen(param_min_path.c_str(), "wb");
    for (const auto &block : program->Blocks()) {
        for (const auto &var_desc : block->Vars()) {
            if (var_desc->Persistable()) {
                if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                    continue;
                }
                LoadWithDumpForInt8(*var_desc, &data, out_file, quantification_fold);
            }
        }
    }
    fclose(out_file);
    delete origin_data;
}

void quantificate_seperated_int8(const std::string model_dir, const std::string param_min_path, int quantification_fold) {
    auto program = loadParams(model_dir + "/__model__");

    std::string shell_command = "mkdir " + param_min_path;
    system(shell_command.c_str());

    for (const auto &block : program->Blocks()) {
        for (const auto &var_desc : block->Vars()) {
            if (var_desc->Persistable()) {
                if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                    continue;
                }
                std::string file_name = param_min_path + "/" + var_desc->Name();
                FILE *out_file = fopen(file_name.c_str(), "wb");
                char *origin_data = Get_binary_data(model_dir + "/" + var_desc->Name());
                char *data = origin_data;
                LoadWithDumpForInt8(*var_desc, &data, out_file, quantification_fold);
                delete origin_data;
                fclose(out_file);
            }
        }
    }
}

void LoadWithDumpForFloat32(const paddle_mobile::framework::VarDesc &var_desc, char **dataP, FILE *out_file, int quantification_fold) {
    // 1. version
    uint32_t version = *reinterpret_cast<uint32_t *>(*dataP);

    // write version
    fwrite(&version, kSize32, 1, out_file);

    *dataP += kSize32;

    // 2 Lod information
    auto *lod_level_ptr = new uint64_t();
    memcpy(lod_level_ptr, *dataP, kSize64);

    uint64_t lod_level = 0;
    // write lod Information
    fwrite(&lod_level, kSize64, 1, out_file);
    delete lod_level_ptr;

    *dataP += kSize64;

    for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size = *reinterpret_cast<uint64_t *>(*dataP);
        // write lod size
        fwrite(&size, kSize64, 1, out_file);
        (*dataP) += kSize64;

        std::vector<size_t> tmp(size / sizeof(size_t));
        for (unsigned long &k : tmp) {
            k = *reinterpret_cast<size_t *>(*dataP);
            (*dataP) += sizeof(size_t);
        }
        // write lod size vector
        fwrite(&tmp, sizeof(size_t), tmp.size(), out_file);
    }

    // 3. tensor version
    uint32_t tensor_version = *reinterpret_cast<uint32_t *>(*dataP);
    // write tensor version
    fwrite(&tensor_version, kSize32, 1, out_file);
    (*dataP) += kSize32;

    // 4. tensor desc
    int32_t size = *reinterpret_cast<int32_t *>(*dataP);
    // write tensor desc
    fwrite(&size, sizeof(int32_t), 1, out_file);
    (*dataP) += sizeof(int32_t);

    std::unique_ptr<char[]> buf(new char[size]);
    for (int m = 0; m < size; ++m) {
        buf.get()[m] = (*dataP)[m];
    }

    fwrite(buf.get(), sizeof(char), static_cast<size_t>(size), out_file);
    (*dataP) += (sizeof(char) * size);

    const paddle_mobile::framework::TensorDesc &desc = var_desc.Tensor_desc();
    int memory_size = 1;
    for (auto l : desc.Dims()) {
        memory_size *= l;
    }

    void *memory = nullptr;
    int type_size = 0;
    switch (desc.DataType()) {
        case paddle_mobile::framework::VARTYPE_TYPE_FP16:
            type_size = 2;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_FP32:
            type_size = 4;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_FP64:
            type_size = 8;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_INT32:
            type_size = 4;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_INT64:
            type_size = 8;
            break;
        case paddle_mobile::framework::VARTYPE_TYPE_BOOL:
            type_size = 1;
            break;
        default:
            break;
    }
    size_t tensorSize = sizeof(char) * memory_size * type_size;

    memory = new char[tensorSize];

    for (int n = 0; n < tensorSize; ++n) {
        static_cast<char *>(memory)[n] = (*dataP)[n];
    }
    *dataP += tensorSize;

    quantification_fold = std::min(std::max(1, memory_size / minimal_fold_size), quantification_fold);
    int step = std::max(memory_size / quantification_fold, 1);

    int visited_fold = 0;
    while (visited_fold * step < memory_size) {
        // for float 32
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::min();

        for (int k = visited_fold * step; k < std::min((visited_fold + 1) * step, memory_size); ++k) {
            min_value = std::min(min_value, static_cast<float *> (memory)[k]);
            max_value = std::max(max_value, static_cast<float *> (memory)[k]);
        }

        float diff = 0.0;
        std::vector<uint8_t> factors;
        for (int g = visited_fold * step; g < std::min((visited_fold + 1) * step, memory_size); ++g) {
            float value = static_cast<float *> (memory)[g];
            auto factor = (uint8_t) round((value - min_value) / (max_value - min_value) * 255);
            factors.push_back(factor);
            float value_quantized = min_value + (factor / 255.0) * (max_value - min_value);
            diff += fabs(value - value_quantized);
            fwrite(&value_quantized, sizeof(float), 1, out_file);
        }
        max_entropy = fmax(max_entropy, entropy(factors));
        if (memory_size > 0) {
            std::cout << "avg diff caused by quantization for var " << var_desc.Name() << " is: " << (diff / memory_size) << std::endl;
        }
        visited_fold++;
    }
}

void
quantificate_combined_float32(const std::string &model_path, const std::string &param_path, const std::string &param_min_path, int quantification_fold) {
    auto program = loadParams(model_path);
    char *origin_data = Get_binary_data(param_path);
    char *data = origin_data;
    FILE *out_file = fopen(param_min_path.c_str(), "wb");
    for (const auto &block : program->Blocks()) {
        for (const auto &var_desc : block->Vars()) {
            if (var_desc->Persistable()) {
                if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                    continue;
                }
                LoadWithDumpForFloat32(*var_desc, &data, out_file, quantification_fold);
            }
        }
    }
    fclose(out_file);
    delete origin_data;
}

void quantificate_seperated_float32(const std::string model_dir, const std::string param_min_path, int quantification_fold) {
    auto program = loadParams(model_dir + "/__model__");

    std::string shell_command = "mkdir " + param_min_path;
    system(shell_command.c_str());

    for (const auto &block : program->Blocks()) {
        for (const auto &var_desc : block->Vars()) {
            if (var_desc->Persistable()) {
                if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                    continue;
                }
                std::string file_name = param_min_path + "/" + var_desc->Name();
                FILE *out_file = fopen(file_name.c_str(), "wb");
                char *origin_data = Get_binary_data(model_dir + "/" + var_desc->Name());
                char *data = origin_data;
                LoadWithDumpForFloat32(*var_desc, &data, out_file, quantification_fold);
                delete origin_data;
                fclose(out_file);
            }
        }
    }
}

int main(int argc, char **argv) {
    const std::string kNoteEg = "( eg:  ./quantify 1 your_combined_model_path output_path  or  ./quantify 0 your_seperated_model_path output_path  or  ./quantify 3 your_seperated_model_path output_path  or  ./quantify 2 your_seperated_model_path output_path)";

    PADDLE_MOBILE_ENFORCE(argc > 1, "wee need params.%s ", kNoteEg.c_str());

    std::string action_type = argv[1];
    PADDLE_MOBILE_ENFORCE(argc > 1 && (action_type) == "0" || action_type == "1" || action_type == "2" || action_type == "3",
                          "only 0, 1, 2 or 3 supported, current is %s %s ",
                          action_type.c_str(),
                          kNoteEg.c_str());

    PADDLE_MOBILE_ENFORCE(argc > 2, "we need your model path. %s ", kNoteEg.c_str());
    std::string base_path = argv[2];

    PADDLE_MOBILE_ENFORCE(argc > 3, "we need your output path. %s ", kNoteEg.c_str());
    std::string output_path = argv[3];

    int quantification_fold = 1;
    if (argc > 4) {
        quantification_fold = std::stoi(argv[4]);
    }

    if (action_type == "0") {
        // for seperated
        const std::string &seperated_min_dir = output_path;
        quantificate_seperated_int8(base_path, seperated_min_dir, quantification_fold);
        return 0;
    }

    if (action_type == "1") {
        // for combined
        const std::string &combined_min_dir = output_path;
        std::string model_path = base_path + "/model";
        std::string param_path = base_path + "/params";
        quantificate_combined_int8(model_path, param_path, combined_min_dir, quantification_fold);
        std::cout << "max entropy : " << max_entropy << std::endl;
        return 0;
    }

    if (action_type == "2") {
        // for seperated
        const std::string &seperated_min_dir = output_path;
        quantificate_seperated_float32(base_path, seperated_min_dir, quantification_fold);
        return 0;
    }

    if (action_type == "3") {
        // for combined
        const std::string &combined_min_dir = output_path;
        std::string model_path = base_path + "/model";
        std::string param_path = base_path + "/params";
        quantificate_combined_float32(model_path, param_path, combined_min_dir, quantification_fold);
        return 0;
    }

    return -1;
}
