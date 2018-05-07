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

#include <iostream>
#include "t_buddy_allocator.h"
namespace paddle_mobile {
    namespace memory {


            BuddyAllocator::~BuddyAllocator() {
                //VLOG(10) << "BuddyAllocator Disconstructor makes sure that all of these "
                //"have actually been freed";
                while (!pool_.empty()) {
                    auto block = static_cast<MemoryBlock*>(std::get<2>(*pool_.begin()));
                    //VLOG(10) << "Free from block (" << block << ", " << max_chunk_size_ << ")";

                    BaseFree(block, max_chunk_size_, block->index());
                    pool_.erase(pool_.begin());
                }
            }
            void* BuddyAllocator::BaseAlloc(size_t* index, size_t size) {
                if (size <= 0) return nullptr;
                *index = 0;  // unlock memory
                void* p;
                //**memsize 32, Bytes
                //if success, return 0;
                posix_memalign(&p, 32ul, size);
                if (p != nullptr) {
                    if (FLAGS_use_pinned_memory) {
                        *index = 1;
                        mlock(p, size);  // lock memory
                    }
                }
                return p;
            }

            void BuddyAllocator::BaseFree(void* p, size_t size, size_t index) {
                if (p != nullptr && index == 1) {
                    munlock(p, size);
                }
                free(p);
            }

            inline size_t align(size_t size, size_t alignment) {
                size_t remaining = size % alignment;
                return remaining == 0 ? size : size + (alignment - remaining);
            }
//这步都返回的是->data()，
            void* BuddyAllocator::Alloc(size_t unaligned_size) {
                // adjust allocation alignment
                size_t size =
                        align(unaligned_size + sizeof(MemoryBlock::Desc), min_chunk_size_);

                // acquire the allocator lock
                std::lock_guard<std::mutex> lock(mutex_);

                //VLOG(10) << "Allocate " << unaligned_size << " bytes from chunk size "
                //<< size;

                // if the allocation is huge, send directly to the system allocator
                if (size > max_chunk_size_) {
                    //VLOG(10) << "Allocate from system allocator.";
                    //return huge memoryblock data()指针。
                    return SystemAlloc(size);
                }

                // query and allocate from the existing chunk
                //get it<2> > size and index 从小到大查找。从旧的内存块中寻找。
                auto it = FindExistChunk(size);

                // refill the pool if failure
                if (it == pool_.end()) {
                    //创建了max_chunk，插入到index=0，将新建的内存指针p指向为memoryblock信息的形式，
                    // 生成<index,size,*memory_block>
                    // 返回pool.insert.first
                    it = RefillPool();
                    // if still failure, fail fatally
                    if (it == pool_.end()) {
                        return nullptr;
                    }
                } else {
                    //VLOG(10) << "Allocation from existing memory block " << std::get<2>(*it)
                    //<< " at address "
                    //<< reinterpret_cast<MemoryBlock*>(std::get<2>(*it))->data();
                }
                //pool_'s
                total_used_ += size;
                total_free_ -= size;

                // split the allocation and return data for use
                return reinterpret_cast<MemoryBlock*>(SplitToAlloc(it, size))->data();
            }
            //将p指向的block放入pool池中并标记为freechunk，如果满足一定条件调用systemallocator的free来释放大块内存。
            void BuddyAllocator::Free(void* p) {
                // Point back to metadata
                auto block = static_cast<MemoryBlock*>(p)->metadata();

                // Acquire the allocator lock
                std::lock_guard<std::mutex> lock(mutex_);

                //VLOG(10) << "Free from address " << block;

                if (block->type() == MemoryBlock::HUGE_CHUNK) {
                    //VLOG(10) << "Free directly from system allocator";
                    BaseFree(block, block->total_size(), block->index());

                    return;
                }

                block->mark_as_free();

                total_used_ -= block->total_size();
                total_free_ += block->total_size();

                // Trying to merge the right buddy
                if (block->has_right_buddy()) {
                    //VLOG(10) << "Merging this block " << block << " with its right buddy "
                    //<< block->right_buddy(cache_);

                    auto right_buddy = block->right_buddy();

                    if (right_buddy->type() == MemoryBlock::FREE_CHUNK) {
                        // Take away right buddy from pool
                        pool_.erase(IndexSizeAddress(right_buddy->index(),
                                                     right_buddy->total_size(),
                                                     right_buddy));

                        // merge its right buddy to the block
                        block->merge(right_buddy);
                    }
                }

                // Trying to merge the left buddy
                if (block->has_left_buddy()) {
                    //VLOG(10) << "Merging this block " << block << " with its left buddy "
                    //<< block->left_buddy(cache_);

                    auto left_buddy = block->left_buddy();

                    if (left_buddy->type() == MemoryBlock::FREE_CHUNK) {
                        // Take away right buddy from pool
                        pool_.erase(IndexSizeAddress(left_buddy->index(),
                                                     left_buddy->total_size(), left_buddy));

                        // merge the block to its left buddy
                        left_buddy->merge(block);
                        block = left_buddy;
                    }
                }

                // Dumping this block into pool
                //VLOG(10) << "Inserting free block (" << block << ", "
                //<< block->total_size(cache_) << ")";
                pool_.insert(
                        IndexSizeAddress(block->index(), block->total_size(), block));

                // Clean up if existing too much free memory

                // Prefer freeing fallback allocation first
                //CleanIdleFallBackAlloc();

                // Free normal allocation
                CleanIdleNormalAlloc();
            }

            size_t BuddyAllocator::Used() { return total_used_; }

            void* BuddyAllocator::SystemAlloc(size_t size) {
                size_t index = 0;
                void* p = BaseAlloc(&index, size);

                //VLOG(10) << "Allocated " << p << " from system allocator.";

                if (p == nullptr) return nullptr;

                static_cast<MemoryBlock*>(p)->init(MemoryBlock::HUGE_CHUNK, index,
                                                   size, nullptr, nullptr);

                return static_cast<MemoryBlock*>(p)->data();
            }

            BuddyAllocator::PoolSet::iterator BuddyAllocator::RefillPool() {

                // Allocate a new maximum sized block
                size_t index = 0;
                void* p = BaseAlloc(&index, max_chunk_size_);

                if (p == nullptr) return pool_.end();

                //VLOG(10) << "Creating and inserting new block " << p
                //<< " from system allocator";
                //新申请的p变成了MemoryBlock信息的形式//??
                static_cast<MemoryBlock*>(p)->init(MemoryBlock::FREE_CHUNK, index,
                                                   max_chunk_size_, nullptr, nullptr);

                total_free_ += max_chunk_size_;

                // dump the block into pool
                return pool_.insert(IndexSizeAddress(index, max_chunk_size_, p)).first;
            }

            BuddyAllocator::PoolSet::iterator BuddyAllocator::FindExistChunk(size_t size) {
                size_t index = 0;

                while (1) {
                    //??? 找到pool里大于等于index，大于等于size的，如果pool_<2>小于size，it就往下窜一个。
                    auto it = pool_.lower_bound(IndexSizeAddress(index, size, nullptr));

                    // no match chunk memory
                    if (it == pool_.end()) return it;

                    if (std::get<0>(*it) > index) {
                        // find suitable one
                        if (std::get<1>(*it) >= size) {
                            return it;
                        }
                        // update and continue
                        // if = index ?
                        index = std::get<0>(*it);
                        continue;
                    }
                    return it;
                }
            }

            void* BuddyAllocator::SplitToAlloc(BuddyAllocator::PoolSet::iterator it,
                                               size_t size) {
                auto block = static_cast<MemoryBlock*>(std::get<2>(*it));
                pool_.erase(it);

                //VLOG(10) << "Split block (" << block << ", " << block->total_size(cache_)
                //<< ") into";
                block->split(size);

                //VLOG(10) << "Left block (" << block << ", " << block->total_size(cache_)
                //<< ")";
                block->set_type(MemoryBlock::ARENA_CHUNK);

                // the rest of memory if exist
                if (block->has_right_buddy()) {
                    if (block->right_buddy()->type() == MemoryBlock::FREE_CHUNK) {
                        //VLOG(10) << "Insert right block (" << block->right_buddy(cache_) << ", "
                        //<< block->right_buddy(cache_)->total_size(cache_) << ")";

                        pool_.insert(
                                IndexSizeAddress(block->right_buddy()->index(),
                                                 block->right_buddy()->total_size(),
                                                 block->right_buddy()));
                    }
                }

                return block;
            }


            void BuddyAllocator::CleanIdleNormalAlloc() {
                auto shall_free_alloc = [&]() -> bool {
                    // free all fallback allocations
                    //if (fallback_alloc_count_ > 0) {
                    //    return true;
                    //}
                    // keep 2x overhead if we haven't fallen back
                    if ((total_used_ + max_chunk_size_) * 2 < total_free_) {
                        return true;
                    }
                    return false;
                };

                if (!shall_free_alloc()) return;

                for (auto pool = pool_.rbegin(); pool != pool_.rend();) {
                    // If free memory block less than max_chunk_size_, return directly
                    if (std::get<1>(*pool) < max_chunk_size_) return;

                    MemoryBlock* block = static_cast<MemoryBlock*>(std::get<2>(*pool));

                    //VLOG(10) << "Return block " << block << " to base allocator.";
                    //？
                    BaseFree(block, max_chunk_size_, block->index());
                    //cache_.invalidate(block);

                    std::cout<<"come in CleanIdelNormalAlloc" << std::endl;
                    pool = PoolSet::reverse_iterator(pool_.erase(std::next(pool).base()));

                    total_free_ -= max_chunk_size_;

                    if (!shall_free_alloc()) return;
                }
            }

    }  // namespace memory
}  // namespace paddle
