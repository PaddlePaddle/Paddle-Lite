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
#include <iostream>

#include "io.h"
#include "common/macro.h"
#include "framework/framework.pb.h"

namespace paddle_mobile {

    void ReadBinaryFile(const std::string& filename, std::string* contents) {
        std::ifstream fin(filename, std::ios::in | std::ios::binary);
        fin.seekg(0, std::ios::end);
        contents->clear();
        contents->resize(fin.tellg());
        fin.seekg(0, std::ios::beg);
        fin.read(&(contents->at(0)), contents->size());
        fin.close();
    }

    template<typename Dtype, Precision P>
    const framework::Program<Dtype, P> Loader<Dtype, P>::Load(const std::string &dirname){
        printf("prediction: %d", P);

        std::string model_filename = dirname + "/__model__";
        std::string program_desc_str;
        ReadBinaryFile(model_filename, &program_desc_str);
        framework::proto::ProgramDesc program_desc_proto;
        program_desc_proto.ParseFromString(program_desc_str);

#ifdef PADDLE_MOBILE_DEBUG
        for (int i = 0; i < program_desc_proto.blocks().size(); ++i) {
            framework::proto::BlockDesc block = program_desc_proto.blocks()[i];
            std::cout << "block: " << block.idx() << std::endl;
            for (int j = 0; j < block.ops().size(); ++j) {
                framework::proto::OpDesc op = block.ops()[j];

                std::cout << " op: " << op.type() << std::endl;
                for (int m = 0; m < op.inputs_size(); ++m) {
                    const framework::proto::OpDesc::Var &var = op.inputs(m);
                    std::cout << "  input parameter: " << var.parameter() << std::endl;
                    for (int n = 0; n < var.arguments().size(); ++n) {
                        std::cout << "   argument - " << var.arguments()[n] << std::endl;
                    }
                }

                for (int y = 0; y < op.outputs_size(); ++y) {
                    const framework::proto::OpDesc::Var &var = op.outputs(y);
                    std::cout << "  output parameter: "<< var.parameter() << std::endl;
                    for (int z = 0; z < var.arguments().size(); ++z) {
                        std::cout << "   argument - " << var.arguments()[z] << std::endl;
                    }
                }

                for (int x = 0; x < op.attrs().size(); ++x) {
                    const framework::proto::OpDesc_Attr attr = op.attrs()[x];
                    std::cout << "  attr name: " << attr.name() << std::endl;
                    std::cout << "  attr type: " << attr.type() << std::endl;

                    switch (attr.type()){
                        case framework::proto::AttrType::BOOLEAN:
                            std::cout << "   boolen: " << attr.b() << std::endl;
                            break;
                        case framework::proto::AttrType::INT:
                            std::cout << "   int: " << attr.i() << std::endl;
                            break;
                        case framework::proto::AttrType::FLOAT:
                            std::cout << "   float: " << attr.f() << std::endl;
                        case framework::proto::AttrType::STRING:
                            std::cout << "   string: " << attr.s() << std::endl;
                        case framework::proto::AttrType::BOOLEANS:
//                            std::vector<bool> bools(attr.bools_size());
                            for (int y = 0; y < attr.bools_size(); ++y) {
                                std::cout << "   bool - " << attr.bools(y) << std::endl;
                            }
                        case framework::proto::AttrType::LONG:
                            std::cout << "   long: " << attr.l() << std::endl;
                        case framework::proto::AttrType::FLOATS:
                            for (int y = 0; y < attr.floats_size(); ++y) {
                                std::cout << "   float - " << y << ": " << attr.floats(y) << std::endl;
                            }
                        case framework::proto::AttrType::INTS:
                            for (int y = 0; y < attr.ints_size(); ++y) {
                                std::cout << "   int - " << y << ": " <<  attr.ints(y) << std::endl;
                            }
                        case framework::proto::AttrType::STRINGS:
                            for (int y = 0; y < attr.strings_size(); ++y) {
                                std::cout << "   string - " << y << ": " << attr.strings(y) << std::endl;
                            }
                    }

                }

            }
            for (int k = 0; k < block.vars().size(); ++k) {
                framework::proto::VarDesc var = block.vars()[k];
                std::cout << " var : " << var.name() << std::endl;
            }
        }

#endif

        framework::Program<Dtype, P> p;
        return  p;
    }

    template class Loader<ARM, Precision::FP32>;
}

