

#include "io/paddle_mobile.h"
#include <cstdlib>
using std::string;

static const std::string g_googlenet_combine = "../models/googlenet_combine";
static const std::string g_googlenet = "../models/googlenet";
using paddle_mobile::Executor;
using paddle_mobile::framework::Program;

    char *Get_binary_data(std::string filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        PADDLE_MOBILE_ENFORCE(file != nullptr, "can't open file: %s ",
                              filename.c_str());
        fseek(file, 0, SEEK_END);
        int64_t size = ftell(file);
        PADDLE_MOBILE_ENFORCE(size > 0, "size is too small");
        rewind(file);
        char *data = new char[size];
        size_t bytes_read = fread(data, 1, size, file);
        PADDLE_MOBILE_ENFORCE(bytes_read == size,
                              "read binary file bytes do not match with fseek");
        DLOG << "Get_binary_data end";
        fclose(file);
        return data;
    }

    void LoadWithDump(const paddle_mobile::framework::VarDesc var_desc,
                    paddle_mobile::framework::LoDTensor *tensor, char **data, FILE *out_file) {
        // 1. version
        uint32_t version = *reinterpret_cast<uint32_t *>(*data);
        // write version
        fwrite(&version, sizeof(uint32_t), 1, out_file );
        (*data) += sizeof(uint32_t);
        // 2 Lod information
        uint64_t *lod_level_ptr = new uint64_t();
        memcpy(lod_level_ptr, (*data), sizeof(uint64_t));
        uint64_t lod_level = 0;
        // write lod Information
        fwrite(&lod_level, sizeof(uint64_t), 1, out_file);
        delete lod_level_ptr;
        (*data) += sizeof(uint64_t);
        auto &lod = *tensor->mutable_lod();
        lod.resize(lod_level);
        for (uint64_t i = 0; i < lod_level; ++i) {
            uint64_t size = *reinterpret_cast<uint64_t *>(*data);
            // write lod size
            fwrite(&size, sizeof(uint64_t), 1, out_file);
            (*data) += sizeof(uint64_t);
            std::vector<size_t> tmp(size / sizeof(size_t));
            for (int k = 0; k < tmp.size(); ++k) {
                tmp[k] = *reinterpret_cast<size_t *>(*data);
                (*data) += sizeof(size_t);
            }
            // write lod size vector
            fwrite(&tmp, sizeof(size_t), tmp.size(), out_file );

            lod[i] = tmp;
        }

        // 3. tensor version
        uint32_t tensor_version = *reinterpret_cast<uint32_t *>(*data);
        // write tensor version
        fwrite(&tensor_version, sizeof(uint32_t), 1, out_file);
        (*data) += sizeof(uint32_t);

        // 4. tensor desc
        int32_t size = *reinterpret_cast<int32_t *>(*data);
        // write tensor desc
        fwrite(&size, sizeof(int32_t), 1, out_file);
        (*data) += sizeof(int32_t);

        std::unique_ptr<char[]> buf(new char[size]);
        for (int m = 0; m < size; ++m) {
            buf.get()[m] = (*data)[m];
        }
        fwrite(buf.get(), sizeof(char), size, out_file);
        (*data) += (sizeof(char) * size);

        const paddle_mobile::framework::TensorDesc &desc = var_desc.Tensor_desc();
        int memory_size = 1;
        for (auto l : desc.Dims()) {
            memory_size *= l;
        }
        tensor->Resize(paddle_mobile::framework::make_ddim(desc.Dims()));

        void *memory = tensor;
        int type_size = 0;
        switch (desc.DataType()) {
            case paddle_mobile::framework::VARTYPE_TYPE_FP16:
                type_size = 2;
                break;
            case paddle_mobile::framework::VARTYPE_TYPE_FP32:
                type_size = 4;
                memory = tensor->mutable_data<float>();
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
        for (int n = 0; n < memory_size * type_size; ++n) {
            static_cast<char *>(memory)[n] = (*data)[n];
        }
        (*data) += (sizeof(char) * memory_size * type_size);
        // for float 32
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::min();
        for (int k = 0; k < memory_size; ++k) {
            min_value = std::min(min_value, static_cast<float *> (memory)[k]);
            max_value = std::max(max_value, static_cast<float *> (memory)[k]);
        }
        fwrite(&min_value, sizeof(float), 1, out_file);
        fwrite(&max_value, sizeof(float), 1, out_file);
        for (int g = 0; g < memory_size; ++g) {
            float value = static_cast<float *> (memory)[g];
            uint8_t factor = (uint8_t) round((value - min_value) / (max_value - min_value) * 255);
            fwrite(&factor, sizeof(uint8_t), 1, out_file);
        }


    }

    void quantificate_combined(std::string model_path, std::string param_path, std::string param_min_path){
        paddle_mobile::Loader<paddle_mobile::CPU,paddle_mobile::Precision::FP32 > loader;
        bool optimize = true;
        auto program = loader.Load(model_path, param_path, optimize);
        char *origin_data = Get_binary_data(program.para_path);
        char *data = origin_data;
        FILE *out_file = fopen(param_min_path.c_str(), "wb");
        for (const auto &block : program.originProgram->Blocks()) {
            for (const auto &var_desc : block->Vars()) {
                auto var = program.scope->Var(var_desc->Name());
                if(var_desc ->Persistable()) {
                    auto tensor = var->template GetMutable<paddle_mobile::framework::LoDTensor>();
                    if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                        continue;
                    }
                    LoadWithDump(*var_desc, tensor, &data,out_file);
                }
            }
        }
        fclose(out_file);
        delete origin_data;

    }
    void quantificate_seperated(std::string model_dir, std::string param_min_path) {
        paddle_mobile::Loader<paddle_mobile::CPU,paddle_mobile::Precision::FP32 > loader;
        bool optimize = true;
        auto program = loader.Load(model_dir, optimize);
        std::string shell_command = "mkdir "+param_min_path;
        system(shell_command.c_str());
        for (const auto &block : program.originProgram->Blocks()) {
            for (const auto &var_desc : block->Vars()) {
                auto var = program.scope->Var(var_desc->Name());
                if(var_desc ->Persistable()) {
                    auto tensor = var->template GetMutable<paddle_mobile::framework::LoDTensor>();
                    if (var_desc->Name() == "feed" || var_desc->Name() == "fetch") {
                        continue;
                    }
                    std::string file_name = param_min_path +"/"+ var_desc->Name();

                    FILE *out_file = fopen(file_name.c_str(), "wb");
                    char *origin_data =
                            Get_binary_data(program.model_path + "/" + var_desc->Name());
                    char *data = origin_data;
                    LoadWithDump(*var_desc, tensor, &data,out_file);
                    delete origin_data;
                    fclose(out_file);
                }
            }
        }

    }
    int main() {
        std::string filename = "params_min";
        std::string model_path = g_googlenet_combine + "/model";
        std::string param_path = g_googlenet_combine + "/params";
        std::string dirname = "param_min_dir";
        std::string model_dir = g_googlenet;
//        quantificate_combined(model_path, param_path,filename);
        quantificate_seperated(model_dir, dirname);

        return 0;
    }






