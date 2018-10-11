#include <cstdio>
#include "export.h"

void export_scope(ProgramPtr program, ScopePtr scope, const std::string & dirname) {
  for (const auto& block: program->Blocks()) {
    for (const auto& var: block->Vars()) {
        if (var->Name() == "feed" || var->Name() == "fetch") {
          continue;
        }
        if (var->Persistable()) {
          auto* v = scope->FindVar(var->Name());
          assert(v != nullptr);
          int count = 1;
          for (auto n: var->Tensor_desc().Dims()) {
            count *= n;
          }

          auto* tensor = v->GetMutable<paddle_mobile::framework::LoDTensor>();
          const float * p = tensor->mutable_data<float>();

          std::string para_file_name = dirname + '/' + var->Name();
          FILE *para_file = fopen(para_file_name.c_str(), "w");
          assert(p != nullptr);
          fwrite(p, sizeof(float), count, para_file);
          fclose(para_file);
          // std::cout << "==> " << var->Name() << " " << count << "\n";
          // for (int i = 0; i < count; i++) {
          //     std::cout << p[i] << ", ";
          // }
          // std::cout << "\n";
        }
    }
  }
}
