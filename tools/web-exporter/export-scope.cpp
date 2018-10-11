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
          const float * p = v->Get<float>();
          int count = 1;
          for (auto n: var->Tensor_desc().Dims()) {
            count *= n;
          }
          std::string para_file_name = dirname + '/' + var->Name();
          FILE *para_file = fopen(para_file_name.c_str(), "w");
          assert(p != nullptr);
          // std::cout << var->Name() << " " << count << "\n";
          fwrite(p, sizeof(float), count, para_file);
          fclose(para_file);
        }
    }
  }
}
