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
#include "t_malloc.h"

namespace paddle_mobile {
    namespace memory {

        BuddyAllocator* GetCPUBuddyAllocator() {
            //static !!
            static BuddyAllocator* a = nullptr;
            if (a == nullptr) {
                a = new BuddyAllocator;
            }
            return a;
        }

        void Copy( void* dst, const void* src, size_t num){
            std::memcpy(dst, src, num);
        };


        void* Alloc(size_t size) {
            //VLOG(10) << "Allocate " << size << " bytes on " << "place";
            //如果调用多次，因为static，所以共享pool_等属性。
            void* p = GetCPUBuddyAllocator()->Alloc(size);
            //VLOG(10) << "  pointer=" << p;
            return p;
        }
        void Free(void* p) {
            //VLOG(10) << "Free pointer=" << p << " on " << "platform::Place(place)";
            GetCPUBuddyAllocator()->Free(p);
        }

        size_t Used() {
            return GetCPUBuddyAllocator()->Used();
        }

        size_t Usage::operator()() const {
            return Used();
        }

        size_t memory_usage() {
            return GetCPUBuddyAllocator()->Used();
        }

    }  // namespace memory
}  // namespace paddle
