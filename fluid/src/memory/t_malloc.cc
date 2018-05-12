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

#include <cstring>
#include <cstdlib>
#include "t_malloc.h"

namespace paddle_mobile {
    namespace memory {
        const int MALLOC_ALIGN = 16;

        void Copy( void* dst, const void* src, size_t num){
            std::memcpy(dst, src, num);
        };


        void* Alloc(size_t size) {
            size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
            char* p = static_cast<char*>(malloc(offset + size));
            if (!p) {
                return nullptr;
            }
            void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
            static_cast<void**>(r)[-1] = p;
            return r;
        }

        void Free(void* ptr) {
            if (ptr){
                free(static_cast<void**>(ptr)[-1]);
            }
        }

    }  // namespace memory
}  // namespace paddle
