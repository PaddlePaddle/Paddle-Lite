
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

#include <vector>

namespace paddle_mobile{
    namespace framework{
        class Shape: public std::vector<int> {
        public:
            Shape(): std::vector<int>(){}

            template <typename Head, typename ...Args>
            Shape(Head head, Args... res){
                InitDims(head, res...);
            }

            Shape operator+(const Shape& shape) {

                Shape tmp_shape(*this);
                int* p = data();
                for (size_t i = 0; i < size(); i++) {
                    tmp_shape[i] = p[i] + shape[i];
                }
                return tmp_shape;
            }

            Shape operator-(const Shape& shape) {

                Shape tmp_shape(*this);
                int* p = data();
                for (size_t i = 0; i < size(); i++) {
                    tmp_shape[i] = p[i] - shape[i];
                }
                return tmp_shape;
            }

            bool operator<(const Shape& shape) const {

                bool flag = size() == shape.size();
                if (!flag) {
                    return false;
                }

                const int* p = data();
                for (size_t i = 0; i < size(); i++) {
                    flag &= (p[i] < shape[i]);
                }
                return flag;
            }

            bool operator<=(const Shape& shape) const{

                bool flag = size() == shape.size();
                if (!flag) {
                    return false;
                }
                const int* p = data();
                for (size_t i = 0; i < size(); i++) {
                    flag &= (p[i] <= shape[i]);
                }
                return flag;
            }

            bool operator==(const Shape& shape) const{

                bool flag = size() == shape.size();
                if (!flag) {
                    return false;
                }
                const int* p = data();
                for (size_t i = 0; i < size(); i++) {
                    flag &= (p[i] == shape[i]);
                }
                return flag;
            }

            int Count(int start = 0) const {
                if (empty()) {
                    return 0;
                }
                int sum = 1;
                for (auto it = begin() + start; it != end(); ++it)
                    sum *= (*it);
                return sum;
            }

            int Dims() const {
                return size();
            }

            bool IsContinue(const Shape &real_shape) const {
                if (real_shape.size() != size()){
                    return false;
                }

                const int* p = data();
                for (int i = size() - 1; i >= 0; i--) {
                    if (p[i] != real_shape[i]) {
                        int size = Count() / Count(i);
                        return size == 1;
                    }
                }
                return true;
            }

            static Shape Zero(int dims){
                Shape shape;
                for (int i = 0; i < dims; ++i) {
                    shape.push_back(0);
                }
                return shape;
            }

            static Shape Minusone(int dims){
                Shape shape;
                for (int i = 0; i < dims; ++i) {
                    shape.push_back(-1);
                }
                return shape;
            }

        private:
            template <typename Head, typename ...Args>
            void InitDims(Head head, Args...args){
                push_back(head);
                InitDims(args...);
            }
            void InitDims(){};
        };
    }
}