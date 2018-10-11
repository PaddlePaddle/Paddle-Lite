#include "export.h"

inline std::string indent(int i) {
  return std::string(i, ' ');
}
void export_nodejs(ProgramPtr program, ScopePtr scope, std::ostream & os) {
  os << "module.exports.program = {\n";
  os << indent(2) << var2str("blocks") << ": [\n";
  for (const auto& block: program->Blocks()) {
    os << indent(4) << "{\n";
    os << indent(6) << var2str("vars") << ": {\n";
    for (const auto& var: block->Vars()) {
      const auto& dim = var->Tensor_desc().Dims();
      os << indent(8) << var2str(var->Name()) << ": {\n";
      os << indent(10) << var2str("dim") << ": " << var2str(dim) << ",\n";
      os << indent(10) << var2str("persistable") << ": " << var2str(var->Persistable()) << "\n";
      os << indent(8) << "},\n";
    }
    os << indent(6) << "},\n";
    os << indent(6) << var2str("ops") << ": [\n";
    for (const auto& op: block->Ops()) {
      os << indent(8) << "{\n";
      os << indent(10) << var2str("type") << ": " << var2str(op->Type()) << ",\n";
      os << indent(10) << var2str("inputs") << ": {\n";
      for (const auto& kv: op->GetInputs()) {
        os << indent(12) << var2str(kv.first) << ": " << var2str(kv.second) << ",\n";
      }
      os << indent(10) << "},\n";

      os << indent(10) << var2str("outputs") << ": {\n";
      for (const auto& kv: op->GetInputs()) {
        os << indent(12) << var2str(kv.first) << ": " << var2str(kv.second) << ",\n";
      }
      os << indent(10) << "},\n";

      os << indent(10) << var2str("attrs") << ": {\n";
      for (const auto& kv: op->GetAttrMap()) {
        os << indent(12) << var2str(kv.first) << ": ";
        os << decltype(kv.second)::ApplyVistor(VarVisitor(), kv.second) << ",\n";
      }
      os << indent(10) << "},\n";
      os << indent(8) << "},\n";
    }
    os << indent(6) << "],\n";
    os << indent(4) << "},\n";
  }
  os << indent(2) << "]\n";
  os << "}\n";
}
