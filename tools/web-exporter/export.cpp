#include "export.h"
#include <sys/stat.h>
#include <sys/types.h>

class FakeExecutor : public paddle_mobile::framework::Executor<paddle_mobile::CPU, paddle_mobile::Precision::FP32> {
public:
  FakeExecutor(const paddle_mobile::framework::Program<paddle_mobile::CPU> p) {
    program_ = p;
    batch_size_ = 1;
    use_optimize_ = true;
    loddable_ = false;
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }
    auto *variable_ptr = program_.scope->Var("batch_size");
    variable_ptr[0].SetValue<int>(1);
    if (program_.combined) {
      InitCombineMemory();
    } else {
      InitMemory();
    }
  }
};

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <combined-modle-dir> <output-dir>\n";
    return -1;
  }
  std::string model_dir = argv[1];
  std::string model_path = model_dir + "/model";
  std::string para_path = model_dir + "/params";

  std::string out_dir = argv[2];
  std::string out_model_js = out_dir + "/model.js";
  std::string out_para_dir = out_dir + "/paras";
  mkdir(out_dir.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
  mkdir(out_para_dir.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);

  std::cout << "loading " << model_path << " & " << para_path << "\n";
  paddle_mobile::framework::Loader<> loader;
  auto program = loader.Load(model_path, para_path, true);
  FakeExecutor executor(program);
  auto optimizedProgram = program.optimizeProgram;
  export_scope(optimizedProgram, program.scope, out_para_dir);
  std::ofstream fs(out_model_js.c_str());
  export_nodejs(optimizedProgram, program.scope, fs);
  fs.close();
  return 0;
}
