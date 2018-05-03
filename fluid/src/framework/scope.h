
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
#include "paddle_mobile_object.h"

#include <unordered_map> //std::unordered_map
#include <list>  //std::list
#include <mutex> //std::mutex
#include "variable.h"

namespace paddle_mobile{
    namespace framework{
        class Scope : public PaddleMobileObject{
        public:
            Scope(){}
            ~Scope(){}


            Scope& NewScope() const;

            /// Create a variable with given name if it doesn't exist.
            Variable* Var(const std::string& name);

            /// Create a variable with a scope-unique name.
            Variable* Var(std::string* name = nullptr);

            void EraseVars(const std::vector<std::string>& var_names);

            /// Find a variable in the scope or any of its ancestors.  Returns
            /// nullptr if cannot find.
            Variable* FindVar(const std::string& name) const;

            const Scope* parent() const { return parent_; }

            /// Find the scope or an ancestor scope that contains the given variable.
            const Scope* FindScope(const Variable* var) const;

            void DeleteScope(Scope* scope) const;

            /// Drop all kids scopes belonged to this scope.
            void DropKids();

            // enumerate all the variables current contains.
            std::vector<std::string> LocalVarNames() const;

            // Rename variable to a new name
            void Rename(const std::string& origin_name,
                        const std::string& new_name) const;

            // Rename variable to a new name and return the new name
            std::string Rename(const std::string& origin_name) const;

            Variable* FindVarLocally(const std::string& name) const;

        private:
            // Call Scope::NewScope for a sub-scope.
            explicit Scope(Scope const* parent) : parent_(parent) {}

            mutable std::unordered_map<std::string, Variable*> vars_;
            mutable std::list<Scope*> kids_;
            Scope const* parent_{nullptr};

            mutable std::mutex mutex_;
        };
    }
}
