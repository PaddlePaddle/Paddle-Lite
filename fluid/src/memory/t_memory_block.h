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

#include <functional>

namespace paddle_mobile{
    namespace memory{
        struct MemoryBlock{
            enum Type{
                FREE_CHUNK,    //memory is free and idle
                ARENA_CHUNK,   //memory is being occupied
                HUGE_CHUNK,    //memory is out of management
                INVALID_CHUNK  //memory is invalid
            };
            void init(Type t, size_t index, size_t size,
                      void* left_buddy, void* right_buddy);

            // All these accessors returns fields in the MemoryBlock::Desc of the memory
            // block.  They all need a MetadataCache instance as their first
            // parameter because they read the MemoryBlock::Desc from the cache.
            Type type() const;
            size_t size() const;
            size_t index() const;
            size_t total_size() const;
            bool has_left_buddy() const;
            bool has_right_buddy() const;
            MemoryBlock* left_buddy() const;
            MemoryBlock* right_buddy() const;

            // Split the allocation into left/right blocks.
            void split(size_t size);

            // Merge left and right blocks together.
            void merge( MemoryBlock* right_buddy);

            // Mark the allocation as free.
            void mark_as_free();

            // Change the type of the allocation.
            void set_type(Type t);

            void* data() const;
            MemoryBlock* metadata() const;


            // MemoryBlock::Desc describes a MemoryBlock.
            struct Desc {
                Desc(MemoryBlock::Type t, size_t i, size_t s, size_t ts,
                     MemoryBlock* l, MemoryBlock* r):
                        type(t), index(i), size(s), total_size(ts),
                        left_buddy(l), right_buddy(r){};
                Desc() :type(INVALID_CHUNK), index(0), size(0), total_size(0),
                        left_buddy(nullptr), right_buddy(nullptr){};

                // Updates guard_begin and guard_end by hashes of the Metadata object.
                //void update_guards();

                // Checks that guard_begin and guard_end are hashes of the Metadata object.
                //bool check_guards() const;

                // TODO(gangliao): compress this
                size_t guard_begin = 0;                 //64bits
                MemoryBlock::Type type; //= MemoryBlock::INVALID_CHUNK; //64bits
                size_t index; //= 0;
                size_t size; //= 0;
                size_t total_size; //= 0;
                MemoryBlock* left_buddy; //= nullptr;   //64bits
                MemoryBlock* right_buddy; //= nullptr;
                size_t guard_end = 0;
            };

            MemoryBlock::Desc load(const MemoryBlock* memory_block) const;
            void save(MemoryBlock* memory_block, const MemoryBlock::Desc& meta_data);

        };

    }
}