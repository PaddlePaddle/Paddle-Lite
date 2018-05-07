//
// Created by liuRuiLong on 2018/5/4.
//



#include "program_desc.h"

namespace paddle_mobile{

namespace framework{
    ProgramDesc::ProgramDesc(const proto::ProgramDesc &desc): desc_(desc){
        for(auto &block_desc : *desc_.mutable_blocks()){
            blocks_.emplace_back(new framework::BlockDesc(block_desc));
        }
    }

    std::shared_ptr<BlockDesc> ProgramDesc::Block(size_t idx){
        return blocks_[idx];
    }
}

}