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
#ifndef MDL_EXCEPTION_H
#define MDL_EXCEPTION_H

#include <string>
#include <sstream>
#include <exception>

namespace mdl {
    const std::string exception_prefix = "MDL C++ Exception: ";
    /**
     * custom C++ exception
     */
    struct MDLException: public std::exception {
        std::string message;
        MDLException(const char *detail, const char *file, const int line) {
            std::stringstream ss;
            ss << exception_prefix << "-Custom Exception- ";
            ss << "| [in file] -:- [" << file << "] ";
            ss << "| [on line] -:- [" << line << "] ";
            ss << "| [detail] -:- [" << detail << "].";
            message = ss.str();
        }

        virtual const char *what() const throw() {
            return message.c_str();
        }
    };
};

#define throw_exception(...) {char buffer[1000]; sprintf(buffer, __VA_ARGS__); std::string detail{buffer}; throw MDLException(detail.c_str(), __FILE__, __LINE__);}

#endif
