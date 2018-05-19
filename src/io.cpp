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

#include <fstream>

#include "common/log.h"
#include "framework/framework.pb.h"
#include "framework/lod_tensor.h"
#include "framework/program_desc.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "io.h"

namespace paddle_mobile {

void ReadBinaryFile(const std::string &filename, std::string *contents) {
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    fin.seekg(0, std::ios::end);
    contents->clear();
    contents->resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(contents->at(0)), contents->size());
    fin.close();
}

template <typename Dtype, Precision P>
void Loader<Dtype, P>::LoadVar(framework::LoDTensor *tensor,
                               const std::string &file_path) {
    //        LOG(kLOG_DEBUG) << "  to load " << file_path;
    //  Log(kLOG_DEBUG) << "123";

    std::ifstream is(file_path);

    std::streampos pos = is.tellg(); //   save   current   position
    is.seekg(0, std::ios::end);
    //        LOG(kLOG_DEBUG) << "  file length = " << is.tellg();
    is.seekg(pos); //   restore   saved   position

    // 1. version
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    //        LOG(kLOG_INFO) << "   version: " << version;

    // 2 Lod information
    uint64_t lod_level;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    //        LOG(kLOG_DEBUG) << "   load level: " << lod_level;
    //        LOG(kLOG_DEBUG) << "   lod info: ";
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
        uint64_t size;
        is.read(reinterpret_cast<char *>(&size), sizeof(size));
        std::vector<size_t> tmp(size / sizeof(size_t));
        is.read(reinterpret_cast<char *>(tmp.data()),
                static_cast<std::streamsize>(size));
        for (int j = 0; j < tmp.size(); ++j) {
            LOG(kLOG_DEBUG1) << "    lod - " << tmp[j];
        }
        lod[i] = tmp;
    }

    // 3. tensor version
    uint32_t tensor_version;
    is.read(reinterpret_cast<char *>(&tensor_version), sizeof(tensor_version));
    //  std::cout << "   tensor_version: " << tensor_version << std::endl;

    // 4. tensor desc
    int32_t size;
    is.read(reinterpret_cast<char *>(&size), sizeof(size));
    //  std::cout << "   tensor desc size: " << size << std::endl;
    std::unique_ptr<char[]> buf(new char[size]);
    is.read(reinterpret_cast<char *>(buf.get()), size);

    framework::proto::VarType::TensorDesc desc;
    desc.ParseFromArray(buf.get(), size);

    //  std::cout << "   desc dims size " << desc.dims().size() <<
    //  std::endl;
    int memory_size = 1;
    for (int l = 0; l < desc.dims().size(); ++l) {
        //    std::cout << "    dim " << l << " value: " << desc.dims()[l]
        //    <<
        //    std::endl;
        memory_size *= desc.dims()[l];
    }

    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(desc.dims().size()));
    std::copy(desc.dims().begin(), desc.dims().end(), std::back_inserter(dims));
    tensor->Resize(framework::make_ddim(dims));

    void *memory;
    int type_size = 0;
    //  std::cout << "    desc pre type: ";
    switch (desc.data_type()) {
    case framework::proto::VarType::FP16:
        //      std::cout << "FP16" << std::endl;
        type_size = 2;
        break;
    case framework::proto::VarType::FP32:
        type_size = 4;
        memory = tensor->mutable_data<float>();
        //      std::cout << "FP32" << std::endl;
        break;
    case framework::proto::VarType::FP64:
        type_size = 8;
        //      std::cout << "FP64" << std::endl;
        break;
    case framework::proto::VarType::INT32:
        type_size = 4;
        //      std::cout << "INT32" << std::endl;
        break;
    case framework::proto::VarType::INT64:
        type_size = 8;
        //      std::cout << "INT64" << std::endl;
        break;
    case framework::proto::VarType::BOOL:
        type_size = 1;
        //      std::cout << "BOOL" << std::endl;
        break;
    default:
        break;
        //      std::cout << "    not support" << std::endl;
    }

    //  std::cout << "    malloc size: " << memory_size * type_size <<
    //  std::endl;
    is.read(static_cast<char *>(memory), memory_size * type_size);
    //  std::cout << "    memory: " << memory << std::endl;
    is.close();
};

template <typename Dtype, Precision P>
const framework::Program<Dtype, P>
Loader<Dtype, P>::Load(const std::string &dirname) {
    std::string model_filename = dirname + "/__model__";
    std::string program_desc_str;
    ReadBinaryFile(model_filename, &program_desc_str);
    framework::proto::ProgramDesc program_desc_proto;
    program_desc_proto.ParseFromString(program_desc_str);

    std::shared_ptr<framework::ProgramDesc> originProgramDesc =
        std::make_shared<framework::ProgramDesc>(program_desc_proto);

    framework::Program<Dtype, P> program;
    program.originProgram = originProgramDesc;

    std::shared_ptr<framework::Scope> scope =
        std::make_shared<framework::Scope>();
    program.scope = scope;

    auto block = originProgramDesc->Block(0);

    for (auto block : originProgramDesc->Blocks()) {
        //    std::cout << "for block" << std::endl;
        for (int i = 0; i < block->Vars().size(); ++i) {
            std::shared_ptr<framework::VarDesc> var_desc = block->Vars()[i];
            auto var = scope->Var(var_desc->Name());
            if (var_desc->GetType() == framework::proto::VarType::LOD_TENSOR) {
                if (var_desc->Persistable() &&
                    var_desc->GetType() !=
                        framework::proto::VarType::FEED_MINIBATCH &&
                    var_desc->GetType() !=
                        framework::proto::VarType::FETCH_LIST) {
                    framework::LoDTensor *tensor =
                        var->GetMutable<framework::LoDTensor>();
                    // to load
                    LoadVar(tensor, dirname + "/" + var_desc->Name());
                }
            } else {
                //        std::cout << "非 lod" << std::endl;
            }
        }
    }

#ifdef PADDLE_MOBILE_DEBUG
    for (int i = 0; i < program_desc_proto.blocks().size(); ++i) {
        framework::proto::BlockDesc block = program_desc_proto.blocks()[i];
        LOG(kLOG_DEBUG) << "block: " << block.idx();
        for (int j = 0; j < block.ops().size(); ++j) {
            if (j == 2) {
                break;
            }
            framework::proto::OpDesc op = block.ops()[j];
            LOG(kLOG_DEBUG1) << "op: " << op.type();
            for (int m = 0; m < op.inputs_size(); ++m) {
                const framework::proto::OpDesc::Var &var = op.inputs(m);
                LOG(kLOG_DEBUG2) << "input parameter: " << var.parameter();
                for (int n = 0; n < var.arguments().size(); ++n) {
                    LOG(kLOG_DEBUG3) << "argument - " << var.arguments()[n];
                }
            }

            for (int y = 0; y < op.outputs_size(); ++y) {
                const framework::proto::OpDesc::Var &var = op.outputs(y);
                LOG(kLOG_DEBUG2) << "out parameter: " << var.parameter();
                for (int z = 0; z < var.arguments().size(); ++z) {
                    LOG(kLOG_DEBUG3) << "argument - " << var.arguments()[z];
                }
            }

            for (int x = 0; x < op.attrs().size(); ++x) {
                const framework::proto::OpDesc_Attr attr = op.attrs()[x];
                LOG(kLOG_DEBUG2) << "attr name: " << attr.name();

                switch (attr.type()) {
                case framework::proto::AttrType::BOOLEAN:
                    LOG(kLOG_DEBUG3) << "boolen: " << attr.b();
                    break;
                case framework::proto::AttrType::INT:
                    LOG(kLOG_DEBUG3) << "int: " << attr.i();
                    break;
                case framework::proto::AttrType::FLOAT:
                    LOG(kLOG_DEBUG3) << "float: " << attr.f();
                case framework::proto::AttrType::STRING:
                    LOG(kLOG_DEBUG3) << "string: " << attr.s();
                case framework::proto::AttrType::BOOLEANS:
                    for (int y = 0; y < attr.bools_size(); ++y) {
                        LOG(kLOG_DEBUG3) << "bools: " << attr.bools(y);
                    }
                case framework::proto::AttrType::LONG:
                    LOG(kLOG_DEBUG3) << "long: " << attr.l();
                case framework::proto::AttrType::FLOATS:
                    for (int y = 0; y < attr.floats_size(); ++y) {
                        LOG(kLOG_DEBUG3) << "floats: " << attr.floats(y);
                    }
                case framework::proto::AttrType::INTS:
                    for (int y = 0; y < attr.ints_size(); ++y) {
                        LOG(kLOG_DEBUG3) << "ints: " << attr.ints(y);
                    }
                case framework::proto::AttrType::STRINGS:
                    for (int y = 0; y < attr.strings_size(); ++y) {
                        LOG(kLOG_DEBUG3) << "strings: " << attr.strings(y);
                    }
                }
            }
        }

        for (int k = 0; k < block.vars().size(); ++k) {
            framework::proto::VarDesc var = block.vars()[k];
            if (var.type().type() == framework::proto::VarType::LOD_TENSOR) {
                LOG(kLOG_DEBUG1) << "var name: " << var.name();
                const framework::proto::VarType::TensorDesc &tensor_desc =
                    var.type().lod_tensor().tensor();
                LOG(kLOG_DEBUG2) << "in var tensor desc dims size: "
                                 << tensor_desc.dims().size();
                int memory_size = 1;
                for (int l = 0; l < tensor_desc.dims().size(); ++l) {
                    LOG(kLOG_DEBUG3) << "var tensor desc dim " << l
                                     << " value: " << tensor_desc.dims()[l];
                }
            }

            if (var.persistable() &&
                var.type().type() !=
                    framework::proto::VarType::FEED_MINIBATCH &&
                var.type().type() != framework::proto::VarType::FETCH_LIST) {
                //        std::cout << "  to load " << var.name() <<
                //        std::endl;
                std::string file_path = dirname + "/" + var.name();
                std::ifstream is(file_path);
                std::streampos pos = is.tellg(); //   save   current   position
                is.seekg(0, std::ios::end);
                //        std::cout << "  file length = " << is.tellg() <<
                //        std::endl;
                is.seekg(pos); //   restore   saved   position

                // 1. version
                uint32_t version;
                is.read(reinterpret_cast<char *>(&version), sizeof(version));
                //        std::cout << "   version: " << version <<
                //        std::endl;

                // 2 Lod information
                uint64_t lod_level;
                is.read(reinterpret_cast<char *>(&lod_level),
                        sizeof(lod_level));
                //        std::cout << "   load level: " << lod_level <<
                //        std::endl;
                //        std::cout << "   lod info: " << std::endl;
                for (uint64_t i = 0; i < lod_level; ++i) {
                    uint64_t size;
                    is.read(reinterpret_cast<char *>(&size), sizeof(size));
                    std::vector<size_t> tmp(size / sizeof(size_t));
                    is.read(reinterpret_cast<char *>(tmp.data()),
                            static_cast<std::streamsize>(size));
                    for (int j = 0; j < tmp.size(); ++j) {
                        //            std::cout << "    lod - " << tmp[j] <<
                        //            std::endl;
                    }
                }

                uint32_t tensor_version;
                is.read(reinterpret_cast<char *>(&version), sizeof(version));
                //        std::cout << "   tensor_version: " <<
                //        tensor_version <<
                //        std::endl;

                int32_t size;
                is.read(reinterpret_cast<char *>(&size), sizeof(size));
                //        std::cout << "   tensor desc size: " << size <<
                //        std::endl;
                std::unique_ptr<char[]> buf(new char[size]);
                is.read(reinterpret_cast<char *>(buf.get()), size);

                framework::proto::VarType::TensorDesc desc;
                desc.ParseFromArray(buf.get(), size);

                //        std::cout << "   desc dims size " <<
                //        desc.dims().size() <<
                //        std::endl;
                int memory_size = 1;
                for (int l = 0; l < desc.dims().size(); ++l) {
                    //          std::cout << "    dim " << l << " value: "
                    //          <<
                    //          desc.dims()[l]
                    //                    << std::endl;
                    memory_size *= desc.dims()[l];
                }

                int type_size = 0;
                //        std::cout << "    desc pre type: ";
                switch (desc.data_type()) {
                case framework::proto::VarType::FP16:
                    //            std::cout << "FP16" << std::endl;
                    type_size = 2;
                    break;
                case framework::proto::VarType::FP32:
                    type_size = 4;
                    //            std::cout << "FP32" << std::endl;
                    break;
                case framework::proto::VarType::FP64:
                    type_size = 8;
                    //            std::cout << "FP64" << std::endl;
                    break;
                case framework::proto::VarType::INT32:
                    type_size = 4;
                    //            std::cout << "INT32" << std::endl;
                    break;
                case framework::proto::VarType::INT64:
                    type_size = 8;
                    //            std::cout << "INT64" << std::endl;
                    break;
                case framework::proto::VarType::BOOL:
                    type_size = 1;
                    //            std::cout << "BOOL" << std::endl;
                    break;
                default:
                    break;
                    //            std::cout << "    not support" <<
                    //            std::endl;
                }

                //        std::cout << "    malloc size: " << memory_size *
                //        type_size
                //                  << std::endl;
                void *memory = malloc(memory_size * type_size);
                is.read(static_cast<char *>(memory), memory_size * type_size);
                //        std::cout << "    memory: " << memory <<
                //        std::endl;
                is.close();
            } else {
                //        std::cout << "  *not load "
                //                  << " var : " << var.name() << std::endl;
            }
        }
    }

#endif
    return program;
}

template class Loader<CPU, Precision::FP32>;

} // namespace paddle_mobile
