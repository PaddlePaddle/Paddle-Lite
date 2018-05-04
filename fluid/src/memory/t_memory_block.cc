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
#pragma once

#include "t_memory_block.h"

namespace paddle_mobile{
    namespace memory{

        void MemoryBlock::init(Type t, size_t index, size_t size,
                               void* left_buddy, void* right_buddy) {
            //this-?save?
            save(this, MemoryBlock::Desc(t, index, size - sizeof(MemoryBlock::Desc), size,
                                            static_cast<MemoryBlock*>(left_buddy),
                                            static_cast<MemoryBlock*>(right_buddy)));
        }

        MemoryBlock::Desc MemoryBlock::load(const MemoryBlock* block) const {
            //auto* desc = reinterpret_cast<const MemoryBlock::Desc*>(block);
            //VLOG(10) << "Load MemoryBlock::Desc type=" << desc->type;
            //PADDLE_ASSERT(desc->check_guards());
            return *reinterpret_cast<const MemoryBlock::Desc*>(block);
        }


        void MemoryBlock::save(MemoryBlock* block,
                                 const MemoryBlock::Desc& original_desc) {
            auto desc = original_desc;
            //desc.update_guards();
                //begin 存储？
                *reinterpret_cast<MemoryBlock::Desc*>(block) = desc;
        }

        //need this or direct return Desc.type.
        MemoryBlock::Type MemoryBlock::type() const {
            return load(this).type;
        }

        size_t MemoryBlock::size() const {
            return load(this).size;
        }

        size_t MemoryBlock::index() const {
            return load(this).index;
        }

        size_t MemoryBlock::total_size() const {
            return load(this).total_size;
        }

        bool MemoryBlock::has_left_buddy() const {
            return left_buddy() != nullptr;
        }

        bool MemoryBlock::has_right_buddy() const {
            return right_buddy() != nullptr;
        }

        MemoryBlock* MemoryBlock::left_buddy() const {
            return load(this).left_buddy;
        }

        MemoryBlock* MemoryBlock::right_buddy() const {
            return load(this).right_buddy;
        }

        void MemoryBlock::split(size_t size) {
            // make sure the split fits
            //std::cout<< "mb.cc LINE65: total_size(*cache)" << total_size(*cache)<<std::endl;
            //PADDLE_ASSERT(total_size() >= size);

            // bail out if there is no room for another partition
            if (total_size() - size <= sizeof(MemoryBlock::Desc)) {
                return;
            }

            // find the position of the split
            void* right_partition = reinterpret_cast<uint8_t*>(this) + size;

            size_t remaining_size = total_size() - size;

            // Add the new block as a buddy
            auto metadata = load(this);

            // Write the metadata for the new block
            //相当于链表，先指向后方数据。
            auto new_block_right_buddy = metadata.right_buddy;

            save(static_cast<MemoryBlock*>(right_partition),
                        MemoryBlock::Desc(FREE_CHUNK, index(),
                                          remaining_size - sizeof(MemoryBlock::Desc),
                                          remaining_size, this, new_block_right_buddy));

            metadata.right_buddy = static_cast<MemoryBlock*>(right_partition);
            metadata.size = size - sizeof(MemoryBlock::Desc);
            metadata.total_size = size;

            save(this, metadata);

            // Write metadata for the new block's right buddy
            if (new_block_right_buddy != nullptr) {
                auto buddy_metadata = load(new_block_right_buddy);
                //update new partition as left buddy.
                buddy_metadata.left_buddy = static_cast<MemoryBlock*>(right_partition);

                save(new_block_right_buddy, buddy_metadata);
            }
        }

        void MemoryBlock::merge(MemoryBlock* right_buddy) {
            // only free blocks can be merged
            //PADDLE_ASSERT(type(*cache) == FREE_CHUNK);
            //PADDLE_ASSERT(right_buddy->type(*cache) == FREE_CHUNK);

            auto metadata = load(this);

            // link this->buddy's buddy
            metadata.right_buddy = right_buddy->right_buddy();

            // link buddy's buddy -> this
            if (metadata.right_buddy != nullptr) {
                auto buddy_metadata = load(metadata.right_buddy);

                buddy_metadata.left_buddy = this;

                save(metadata.right_buddy, buddy_metadata);
            }

            metadata.size += right_buddy->total_size();
            metadata.total_size += right_buddy->total_size();

            save(this, metadata);
            //desc还是占用地方？

            save(right_buddy,
                        MemoryBlock::Desc(INVALID_CHUNK, 0, 0, 0, nullptr, nullptr));
        }

        void MemoryBlock::mark_as_free() {
            // check for double free or corruption
            //PADDLE_ASSERT(type(*cache) != FREE_CHUNK);
            //PADDLE_ASSERT(type(*cache) != INVALID_CHUNK);
            set_type(FREE_CHUNK);
        }

        void MemoryBlock::set_type(Type t) {
            auto metadata = load(this);
            metadata.type = t;
            save(this, metadata);
        }
// *desc this 之后的东西 1 void* 可以指向任何
        void* MemoryBlock::data() const {
            return const_cast<MemoryBlock::Desc*>(
                           reinterpret_cast<const MemoryBlock::Desc*>(this)) +
                   1;
        }
// memoryblock * &desc - 1
        MemoryBlock* MemoryBlock::metadata() const {
            return const_cast<MemoryBlock*>(reinterpret_cast<const MemoryBlock*>(
                    reinterpret_cast<const MemoryBlock::Desc*>(this) - 1));
        }

    }
}