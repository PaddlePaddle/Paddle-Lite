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

#include <memory>   // malloc size_t memset memcpy ...
#include "traits.h" // struct DeviceTraits<ARM> ...

namespace paddle_mobile {
    namespace framework {
        const int MALLOC_ALIGN = 16;

        static inline void *FastMalloc(size_t size) {
            size_t offset = sizeof(void *) + MALLOC_ALIGN - 1;
            char *p = static_cast<char *>(malloc(offset + size));
            if (!p) {
                return nullptr;
            }
            void *r = reinterpret_cast<void *>(reinterpret_cast<size_t>(p + offset) & (~(MALLOC_ALIGN - 1)));
            static_cast<void **>(r)[-1] = p;
            return r;
        }

        static inline void FastFree(void *ptr) {
            if (ptr) {
                free(static_cast<void **>(ptr)[-1]);
            }
        }

        template<typename DeviceType, typename target_category = typename DeviceTraits<DeviceType>::target_category>
        struct BaseAllocator {};

        template<typename DeviceType>
        struct BaseAllocator<DeviceType, __host_target> {
            typedef __invalid_type stream_t;

            /**
             * \brief wrapper of memory allocate function, with alignment of 16 bytes
             *
            */
            static void MemAlloc(void **ptr, size_t n) {
                *ptr = (void *) FastMalloc(n);
            }

            /**
             * \brief wrapper of memory free function
             *
            */
            static void MemFree(void *ptr) {
                if (ptr != nullptr) {
                    FastFree(ptr);
                }
            }

            /**
             * \brief wrapper of memory set function, input value only supports 0 or -1
             *
            */
            static void MemSet(void *ptr, int value, size_t n) {
                memset(ptr, value, n);
            }


            /**
             * \brief memory copy function, use memcopy from host to host
             *
            */
            static void SyncMemcpy(void *dst, int dst_id, const void *src, int src_id, \
            size_t count, __HtoH) {
                memcpy(dst, src, count);
                //LOG(INFO) << "host, sync, H2H, size: " << count;
            }

            /**
             * \brief same with sync_memcpy
             * @tparam void
             * @param dst
             * @param dst_id
             * @param src
             * @param src_id
             * @param count
             */
            static void ASyncMemcpy(void *dst, int dst_id, const void *src, int src_id, \
            size_t count, stream_t &stream, __HtoH) {
                memcpy(dst, src, count);
                //LOG(INFO) << "host, sync, H2H, size: " << count;
            }

            /**
             * \brief memcpy peer to peer, for device memory copy between different devices
             * @tparam void
             * @param dst
             * @param dst_dev
             * @param src
             * @param src_dev
             * @param count
             */
            static void SyncMemcpyP2P(void *dst, int dst_dev, const void *src, \
            int src_dev, size_t count) {}

            /**
             * \brief asynchronize memcpy peer to peer, for device memory copy between different devices
             * @tparam void
             * @param dst
             * @param dst_dev
             * @param src
             * @param src_dev
             * @param count
             */
            static void ASyncMemcpyP2P(void *dst, int dst_dev, const void *src, \
            int src_dev, size_t count, stream_t &stream) {}

            /**
             * \brief host target return 0
             * @return      always return 0
             */
            static int GetDeviceID() {
                return 0;
            }
        };
    }
}