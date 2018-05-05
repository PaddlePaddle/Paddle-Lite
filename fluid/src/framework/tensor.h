
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

#include "shape.h"
#include "common/types.h"
#include "traits.h"
#include "tensor_traits.h"
#include "base_allocator.h"
#include "allocator.h"
#include "common.h"
namespace paddle_mobile {
    namespace framework {
         typedef Tensor<ARM, PM_FLOAT, NCHW> Tensor4f;
        class TensorBase: public PaddleMobileObject {
        public:
            TensorBase() {}
            virtual ~TensorBase() {}
            virtual PMStatus set_shape(Shape valid_shape, Shape shape = Shape(), \
        Shape offset = Shape()) = 0;
            virtual PMStatus reshape(Shape valid_shape, Shape shape = Shape(), \
        Shape offset = Shape()) = 0;
            virtual PMStatus re_alloc(Shape shape) = 0;
            virtual bool is_continue_mem() const = 0;
            virtual int size() const = 0;
            virtual int valid_size() const = 0;
            virtual int count(int start, int end) const = 0;
            virtual int count_valid(int start, int end) const = 0;
            virtual int dims() const = 0;
            virtual Shape shape() const = 0;
            virtual Shape valid_shape() const = 0;
            virtual Shape get_stride() const = 0;
            virtual Shape offset() const = 0;
            virtual int device_id() const = 0;
            virtual int num() const = 0;
            virtual int num_index() const = 0;
            virtual int channel() const = 0;
            virtual int channel_index() const = 0;
            virtual int height() const = 0;
            virtual int height_index() const = 0;
            virtual int width() const = 0;
            virtual int width_index() const = 0;
        };

        template<typename DeviceType, DataType datatype, typename LayOutType = NCHW>
        class Tensor : public TensorBase {
        public:
            typedef DeviceType targetType_t;
            typedef typename DataTrait<datatype>::dtype Dtype;
            typedef typename DeviceTraits<DeviceType>::target_category target_category;
            typedef typename DeviceTraits<DeviceType>::target_type target_type;
            typedef BaseAllocator<DeviceType> API;
            typedef TensorTraits<Tensor<DeviceType, datatype, LayOutType> > TensorAPI;
            typedef typename TensorAPI::layout_category layout_category;
            typedef typename TensorAPI::layout_type layout_type;

            /**
             * \brief default constructor
             */
            Tensor() {
                shape_ = Shape::Zero(TensorAPI::layout_dims::value);
                valid_shape_ = Shape::Zero(TensorAPI::layout_dims::value);
                offset_ = Shape::Zero(TensorAPI::layout_dims::value);
                buf_ = std::make_shared<Allocator<DeviceType> >();
                is_subbuf_ = false;
            }

            /**
             * \brief constructor with shape, memory is alloced according to shape
             */
            Tensor(Shape shape) {
                shape_ = shape;
                valid_shape_ = shape;
                offset_ = Shape::Zero(shape.Dims());
                buf_ = std::make_shared<Allocator<DeviceType> >(shape.Count() * type_len_);
                is_subbuf_ = false;
            }

            /**
             * \brief copy constructor, shallow copy
             */
            Tensor(const Tensor<DeviceType, datatype, LayOutType>& tensor){
                shape_ = tensor.shape_;
                valid_shape_ = tensor.valid_shape_;
                offset_ = tensor.offset_;
                buf_  = tensor.buf_;
                is_subbuf_ = tensor. is_subbuf_;
            }

            /**
             * \brief only change the shape and valid shape, do nothing to memory
             * @param shape
             * @param valid_shape
             * @param offset
             */
            PMStatus set_shape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {

                if (shape.Dims() != TensorAPI::layout_dims::value || \
            valid_shape.Dims() != TensorAPI::layout_dims::value \
            || offset.Dims() != TensorAPI::layout_dims::value || \
            !(valid_shape > Shape::Zero(TensorAPI::layout_dims::value))) { \
            return PMInvalidValue; \
        }
                valid_shape_ = valid_shape;

                if (!is_subbuf_) {
                    if (shape_.Count() <= valid_shape_.Count()) {
                        shape_ = valid_shape_;
                    }
                    offset_ = Shape::Zero(TensorAPI::layout_dims::value);
                } else {
                    auto shape_zero = Shape::Zero(TensorAPI::layout_dims::value);
                    if (shape_ == shape_zero) {
                        shape_ = valid_shape;
                    }
                    if (!(valid_shape_ + offset_ <= shape_)) {
                        \
                return PMInvalidValue; \

                    }
                }
                return PMSuccess;
            }

            /**
             * \brief free old buffer and alloc a new tensor buffer
             */
            PMStatus re_alloc(Shape shape){
                if (!shape.Dims() == TensorAPI::layout_dims::value) {
                     return PMInvalidValue;
                }
                if (is_subbuf_ || is_shared_) {
                    return PMOutOfAuthority;
                }

                shape_ = shape;
                valid_shape_ = shape_;
                offset_ = Shape::Zero(shape_.Dims());
                buf_->Alloc(shape_.Count() *type_len_);
                return PMSuccess;
            }

            /**
             * \brief change tensor shape,
             * if input shape's count is bigger than the capacity of buffer, alloc a new buffer
             * @param shape
             */
            PMStatus reshape(Shape valid_shape, Shape shape = Shape(), Shape offset = Shape()) {
                if (shape.Dims() != TensorAPI::layout_dims::value || \
            valid_shape.Dims() != TensorAPI::layout_dims::value \
            || offset.Dims() != TensorAPI::layout_dims::value || \
            !(valid_shape > Shape::Zero(TensorAPI::layout_dims::value))) { \
            return  PMInvalidValue; \
        }
                valid_shape_ = valid_shape;

                if (! is_subbuf_) {
                    if (shape_.Count() < valid_shape_.Count()) {
                        shape_ = valid_shape_;
                    }
                    offset_ = Shape::Zero(TensorAPI::layout_dims::value);
                } else {
                    if (shape_ == Shape::Zero(TensorAPI::layout_dims::value)) {
                        shape_ = valid_shape;
                    }
                    if (!(valid_shape_ + offset_ <= shape_)) { \
                return  PMInvalidValue; \
            }
                }
                bool exceed_flag = shape_.Count() * type_len_ > buf_->GetCapacity() \
            && ( is_subbuf_ || is_shared_);
                if (exceed_flag) {
                    return  PMOutOfAuthority;
                }
//                CHECK_EQ(exceed_flag, false) << "shared tensor shape exceed origin data buffer size";
                PM_CHECK(buf_->ReAlloc(shape_.Count() * type_len_));
                return PaddleSuccess;
            }

            bool is_continue_mem() const {
            }

            /**
             * \brief return shape count, from start index to end index(end index is excluded)
             * \param start input start index
             * \param end   input end index (exclude in calculation)
             * \return the size from start index to end index
             */
            int count(int start, int end) const {
            }

            /**
             * \brief return valid_shape count, from start index to end index(end index is excluded)
             * \param start input start index
             * \param end   input end index (exclude in calculation)
             * \return the size from start index to end index
             */
            int count_valid(int start, int end) const {
            }

            /**
             * \brief return tensor shape size, not the valid shape size
             */
            int size() const {
                return shape_.Count();
            }

            /**
             * \brief return the valid shape size
             * @return
             */
            int valid_size() const{
            }

            /**
             * \brief return tensor shape dims
             */
            int dims() const {
                return TensorAPI::layout_dims::value;
            }

            /**
             * \brief return tensor shape, entire memory buffer shape
             */
            Shape shape() const{
                return shape_;
            }

            /**
             * \brief return valid shape of tensor
             */
            Shape valid_shape() const {
                return valid_shape_;
            }

            /**
             * \brief compute data stride
             * @return
             */
            Shape get_stride() const {
            }

            /**
             * \brief return tensor offset, which holds the offset in each dim
             */
            Shape offset() const {
                return offset_;
            }

            /**
             * \brief return reference shared_ptr of tensor
             */
            const std::shared_ptr<Allocator<DeviceType> >& get_buf() const {
            }

            /**
             * \brief return tensor device id
             */
            int device_id() const {
            }

            /**
             * \brief return number
             * @return
             */
            int num() const {
                return TensorAPI::num(valid_shape_);
            }

            /**
             * \brief return number index in shape
             * @return
             */
            int num_index() const {
                return TensorAPI::num_idx::value;
            };

            /**
             * \brief return channel
             * @return
             */
            int channel() const {
                return TensorAPI::channel(valid_shape_);
            }

            /**
             * \brief return channel index in shape
             * @return
             */
            int channel_index() const {
                return TensorAPI::channel_idx::value;
            }

            /**
             * \brief return height
             * @return
             */
            int height() const {
                return TensorAPI::height(valid_shape_);
            }

            /**
             * \brief return height index in shape
             * @return
             */
            int height_index() const {
                return TensorAPI::height_idx::value;
            }

            /**
             * \brief return width
             * @return
             */
            int width() const {
            }

            /**
             * \brief return height index in shape
             * @return
             */
            int width_index() const {
                return TensorAPI::width_idx::value;
            }

            /**
             * \brief return tensor mutable data pointer, with data type of current tensor (Dtype*)
             */
            Dtype* mutable_data(int index = 0) {
            }

            /**
             * \brief return tensor data pointer, with data type of current tensor (Dtype*)
             */
            const Dtype * data(int index = 0) const {
            }

            /**
             * \brief share from same layout_type and same date type tensor,
             * if shared tensor target is the same with current tensor target, buffer is shared
             * otherwise, tensor buffer is deep copied
             * \details only shared buffer ptr, current tensor will have continuous memory,
             * only if current shape and valid shape are the same, and offset is all set to 0
             */
            //template <typename Tensor1,
            //    class = typename std::enable_if<std::is_same<layout_type, typename TensorTraits<Tensor1>::layout_type>::value>::type>
            //class = typename std::enable_if<std::is_same<layout_type, typename TensorTraits<Tensor1>::layout_type>::value>::type >
            template <typename Tensor_t>
            PMStatus share_from(const Tensor_t& tensor) {

            }

            /**
             * \brief Deep copy data within region of interest from input tensor
             */
            template <typename DeviceType_t, typename LayOutType_t>
            PMStatus copy_from(const Tensor<DeviceType_t, datatype, LayOutType_t>& tensor) {
            }

            /**
             * \brief asynchronously copy entire buffer from source tensor
             */
            template <typename DeviceType_t, typename LayOutType_t, typename stream_type \
        = typename IF<std::is_same<typename DeviceTraits<DeviceType>::target_category, __host_target>::value, \
        typename BaseAllocator<DeviceType_t>::stream_t, typename BaseAllocator<DeviceType>::stream_t>::Type>
            PMStatus async_copy_from(const Tensor<DeviceType_t, datatype, LayOutType_t>& tensor, \
        stream_type stream) {
            }

        


        private:
            //! length of datatype
            size_t type_len_ = sizeof(Dtype);
            //! \brief represent the raw mem shape
            Shape shape_;
            //! \brief represent the mem you have right to access shape
            Shape valid_shape_;
            //! \brief represent the offset idx between shape_ and _real_shape
            Shape offset_;
            //! buffer shared ptr, hold the data pointer, and buffer capacity
            std::shared_ptr<Allocator<DeviceType> > buf_ = nullptr;
            //! \brief events tree, to synchronize the tensor 暂时不用
//            EventsTree<DeviceType> _events_tree;
            //! \brief share sub-buffer flag
            bool is_subbuf_ = false;
            bool is_shared_ = false;

            //! \brief get data real start index
            int start_index() const;
        };
    } // namespace framework
} // namespace paddle_mobile




