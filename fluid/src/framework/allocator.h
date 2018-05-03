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

#include "base_allocator.h"
#include "paddle_mobile_object.h"

namespace paddle_mobile{
    namespace framework{
        template <typename DeviceType>
        class Allocator : public PaddleMobileObject{
        public:
            typedef BaseAllocator<DeviceType> BaseAlloc;
            Allocator() :
                    data_(nullptr), own_data_(true), count_(0), capacity_(0),id_(0){}

            explicit Allocator(size_t size):
                    data_(nullptr), own_data_(true), count_(size), capacity_(size),id_(0) {
                Alloc(size);
            }

            explicit Allocator(void* data, size_t size, int id = 0):
                    own_data_(false), count_(size), capacity_(size), id_(0){
                data_ = data;
                if(id_ != id){}
            }

            Allocator(Allocator<DeviceType>& allocator){
                count_ = allocator.count_;
                id_ = allocator.id_;
                if(allocator.id_ == id_) {
                    data_ = allocator.data_;
                    own_data_ = false;
                    capacity_ = count_;
                } else{
                    own_data_ = true;
                    ReAlloc(allocator.count_);
                    BaseAlloc::SyncMemcpyP2P(data_, id_, allocator.GetData(), allocator.id_, allocator.count_);
                }
            }
            Allocator& operator = (Allocator<DeviceType>& allocator){
                this->count_ = allocator.count_;
                this->id_ = BaseAlloc::GetDeviceID();
                if(allocator.id_ == this ->id_){
                    this->data_ = allocator.data_;
                    this->capacity_ = this->count_;
                    this->own_data_ = false;
                }else{
                    this->own_data_ = true;
                    this->ReAlloc(allocator.count_);
                    BaseAlloc::SyncMemcpyP2P(this->data_, this->id_, allocator.GetData(),
                    allocator.id_, allocator.count_);
                }
                return *this;
            }

            int SharedFrom(Allocator<DeviceType>& allocator){
                count_ = allocator.count_;
                id_ = BaseAlloc::GetDeviceID();
                if(allocator.id_ == id_){
                    data_ = allocator.data_;
                    capacity_ = count_;
                    own_data_ = false;
                    return 1;
                } else {
                    own_data_ = true;
                    ReAlloc(allocator.count_);
                    BaseAlloc::SyncMemcpyP2P(data_,id_, allocator.GetData(), allocator.id_,
                    allocator.count_);
                    return 0;
                }
            }

            ~Allocator(){
                Clean();
            }

            void MemSet(int c, size_t size){
                if(!own_data_ || count_ < size){
                    return;
                }
                BaseAlloc::MemSet(data_, c, size);
            }

            void ReAlloc(size_t size){
                if (size > capacity_ || data_ == nullptr){
                    if (own_data_) {
                        Clean();
                        BaseAlloc::MemAlloc(&data_, size);
                        capacity_ = size;
                    } else{
                        return;
                    }
                }
                count_ = size;
            }

            void Alloc(size_t size){
                Clean();
                BaseAlloc::MemAlloc(&data_, size);
                capacity_ = size;
                own_data_ = true;
                count_ = size;
            }

            int GetID() const {
                return id_;
            }


            /**
             * \brief synchronously copy from other Buf
             */
            //todo
            template <typename DeviceType_t>
            void SyncCopyFrom(Allocator<DeviceType_t>& allocator){

            }

            const void* GetData(){
                return data_;
            }

            void* GetMutableData(){
                return data_;
            }

            inline size_t GetCount() const{
                return count_;
            }

            inline size_t GetCapacity() const {
                return capacity_;
            }

        private:
            int id_; //todo if support multi devices
            void* data_;
            bool own_data_;
            size_t count_;
            size_t capacity_;

            void Clean(){
                if(own_data_ && capacity_ > 0){
                    count_ = 0;
                    capacity_ = 0;
                    own_data_ = false;
                    BaseAlloc::MemFree(data_);
                }
                data_ = nullptr;
            }
        };
    }
}