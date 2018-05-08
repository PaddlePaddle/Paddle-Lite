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

#include <mutex>  // NOLINT
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>

#define FLAGS_use_pinned_memory 1
namespace paddle_mobile {
    namespace memory {
            class BuddyAllocator {
            public:
                BuddyAllocator() = default;
                ~BuddyAllocator();

            public:
                void* Alloc(size_t unaligned_size);
                void Free(void* ptr);
                size_t Used();

            public:
                // Disable copy and assignment
                BuddyAllocator(const BuddyAllocator&) = delete;
                BuddyAllocator& operator=(const BuddyAllocator&) = delete;

            private:
                // Tuple (allocator index, memory size, memory address)
                using IndexSizeAddress = std::tuple<size_t, size_t, void*>;
                // Each element in PoolSet is a free allocation
                using PoolSet = std::set<IndexSizeAddress>;

                /*! \brief Allocate fixed-size memory from system */
                void* SystemAlloc(size_t size);

                /*! \brief If existing chunks are not suitable, refill pool */
                PoolSet::iterator RefillPool();

                /**
                 *  \brief   Find the suitable chunk from existing pool and split
                 *           it to left and right buddies
                 *
                 *  \param   it     the iterator of pool list
                 *  \param   size   the size of allocation
                 *
                 *  \return  the left buddy address
                 */
                void* SplitToAlloc(PoolSet::iterator it, size_t size);

                /*! \brief Find the existing chunk which used to allocation */
                PoolSet::iterator FindExistChunk(size_t size);

                /*! \brief Clean idle fallback allocation */
                //void CleanIdleFallBackAlloc();

                /*! \brief Clean idle normal allocation */
                void CleanIdleNormalAlloc();



            private:
                size_t total_used_ = 0;  // the total size of used memory
                size_t total_free_ = 0;  // the total size of free memory

                const size_t cpu_max_size = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
                const size_t min_chunk_size_ = 1 << 12;  // the minimum size of each chunk
                const size_t max_chunk_size_ = cpu_max_size / 32;  // the maximum size of each chunk
            private:
                /**
                 * \brief A list of free allocation
                 *
                 * \note  Only store free chunk memory in pool
                 */
                PoolSet pool_;

                /*! Record fallback allocation count for auto-scaling */
                size_t fallback_alloc_count_ = 0;

            private:
                /*! Unify the metadata format between GPU and CPU allocations */

            private:
                /*! Allocate CPU/GPU memory from system */
                std::mutex mutex_;

            private:

                void* BaseAlloc(size_t* index, size_t size);
                void  BaseFree(void* p, size_t size, size_t index);
            };

    }  // namespace memory
}  // namespace paddle
