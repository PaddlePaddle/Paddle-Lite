#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include "paddle_api.h"  // NOLINT

#define IPTCORE_PADDLE_MOBILE
#define IPTCORE_PADDLE_BENCHMARK
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_full_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
#ifdef IPTCORE_PADDLE_MOBILE
#else
#ifdef _WIN32
#include "paddle_use_kernels.h"  // NOLINT
#include "paddle_use_ops.h"      // NOLINT
#endif
#endif

#ifdef IPTCORE_PADDLE_BENCHMARK
class Timer {
private:
    std::chrono::high_resolution_clock::time_point inTime, outTime;

public:
    void startTimer() { inTime = std::chrono::high_resolution_clock::now(); }

    // unit millisecond
    float getCostTimer() {
        outTime = std::chrono::high_resolution_clock::now();
        return static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(outTime - inTime)
                .count() /
                1e+3);
    }
};
#endif

template<typename T>
double compute_mean(const T* in, const size_t length) {
    double sum = 0.;
    for (size_t i = 0; i < length; ++i) {
        sum += in[i];
    }
    return sum / length;
}

template<typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
    if (!has_mean) {
        mean = compute_mean<T>(in, length);
    }

    double variance = 0.;
    for (size_t i = 0; i < length; ++i) {
        variance += pow((in[i] - mean), 2);
    }
    variance /= length;
    return sqrt(variance);
}

int64_t shape_production(const paddle::lite_api::shape_t& shape) {
    int64_t res = 1;
    for (auto i : shape) {
        res *= i;
    }
    return res;
}

class InputData {
public:
    int _type = -1; ///int32, int64, float32
    bool _lod = false;
    std::vector<int64_t> _shape;
    std::vector<int32_t> _int32_data;
    std::vector<int64_t> _int64_data;
    std::vector<float> _float32_data;
    std::vector<std::vector<uint64_t>> _lod_data = {{0, 1}, {0, 1}};
};

class UserPersonaInfer {
public:
#ifdef IPTCORE_PADDLE_MOBILE
    void create_paddle_light_predictor(const std::string& model_file);
#else
    void create_paddle_full_predictor(const std::string& model_dir);
#endif
    void prepare(const std::string& path);
    void infer();
private:
    void infer_specific_item(paddle::lite_api::PaddlePredictor *predictor);
    std::shared_ptr<paddle::lite_api::PaddlePredictor> _paddle_predictor;
    std::vector<std::map<std::string, InputData> > _batch;
};

#ifdef IPTCORE_PADDLE_MOBILE
void UserPersonaInfer::create_paddle_light_predictor(const std::string& model_file) {
    // 1. Set MobileConfig
    paddle::lite_api::MobileConfig config;
    config.set_model_from_file(model_file);
    config.set_power_mode(paddle::lite_api::LITE_POWER_HIGH);
    // 2. Create PaddlePredictor by MobileConfig
    _paddle_predictor =
        paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);
}
#else
void UserPersonaInfer::create_paddle_full_predictor(const std::string& model_dir) {
    // 1. Create CxxConfig
    paddle::lite_api::CxxConfig config;
    config.set_model_dir(model_dir);
    config.set_valid_places({paddle::lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                                paddle::lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
    // 2. Create PaddlePredictor by CxxConfig
    _paddle_predictor =
        paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(config);
}
#endif
namespace {
using namespace std;
template <class T>
void extract_num(const string &str, vector<T> &results) {
    stringstream ss;

    /* Storing the whole string into string stream */
    ss << str;

    /* Running loop till the end of the stream */
    string temp;
    T found;
    while (!ss.eof()) {

        /* extracting word by word from stream */
        ss >> temp;

        /* Checking the given word is integer or not */
        if (stringstream(temp) >> found)
            results.emplace_back(found);

        /* To save from space at the end of string */
        temp = "";
    }
}
}

void UserPersonaInfer::prepare(const std::string& path) {
    ///xia_i	186	tgt_generation_mask	float32	(1, 1, 33)	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    std::ifstream in(path.c_str());
    std::string line;
    std::string current_idx;
    while (std::getline(in, line)) {
        if (line.empty()) {
            break;
        }
        if (line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            break;
        }
        std::vector<std::string> strings;
        std::istringstream f(line);
        std::string s;
        while (getline(f, s, '\t')) {
            strings.push_back(s);
        }
        if (current_idx != strings.at(1)) {
            _batch.push_back(std::map<std::string, InputData>());
            current_idx = strings[1];
        }
        if (strings.at(2) == "lods") {
            if (strings.at(3) != "[[0, 1], [0, 1]]") {
                throw std::invalid_argument("invalid lod");
            }
            continue;
        }
        auto& input_data = _batch.back()[strings.at(2)];

        extract_num(strings.at(4), input_data._shape);
        if (strings[0] == "lod_i") {
            input_data._lod = true;
        }
        if (strings.at(3) == "int32") {
            input_data._type = 0;
            extract_num(strings.at(5), input_data._int32_data);
        } else if (strings.at(3) == "int64") {
            input_data._type = 1;
            extract_num(strings.at(5), input_data._int64_data);
        } else if (strings.at(3) == "float32") {
            input_data._type = 2;
            extract_num(strings.at(5), input_data._float32_data);
        } else {
            throw std::invalid_argument("invalid type");
        }

    }
}

void UserPersonaInfer::infer_specific_item(paddle::lite_api::PaddlePredictor *predictor){
    static int count = 0;
    if (_batch.empty()) {
        return;
    }
    auto &inputs = _batch[count];
    auto names = predictor->GetInputNames();
    for (auto &name : names) {
        auto& input = inputs[name];
        auto tensor = predictor->GetInputByName(name);
        tensor->Resize(input._shape);
        if (input._type == 0) {
            auto input_data = tensor->mutable_data<int32_t>();
            std::copy(input._int32_data.begin(), input._int32_data.end(), input_data);
        } else if (input._type == 1) {
            auto input_data = tensor->mutable_data<int64_t>();
            std::copy(input._int64_data.begin(), input._int64_data.end(), input_data);
        } else if (input._type == 2) {
            auto input_data = tensor->mutable_data<float>();
            std::copy(input._float32_data.begin(), input._float32_data.end(), input_data);
        } else {
            throw std::invalid_argument("invalid name");
        }
        if (input._lod) {
            tensor->SetLoD(input._lod_data);
        }
    }

    predictor->Run();

    std::cout << "\n";
    for (int idx = 0; idx != 2; ++idx) {
        auto output_tensor = predictor->GetOutput(idx);
        auto total_size = shape_production(output_tensor->shape());
        std::cout << "xiarj_" << count << "\t";
        for (int i = 0; i < total_size; ++i) {
            if (idx == 0) {
                std::cout << output_tensor->data<int64_t>()[i] << "\t";
            } else {
                std::cout << output_tensor->data<float>()[i] << "\t";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::flush;

    if (++count == _batch.size()){
        count = 0;
    }
}

void UserPersonaInfer::infer() {
    static int idx = 0;
    auto predictor = _paddle_predictor.get();
    if (!predictor) {
        return;
    }
    // 3. Prepare input data

    // 4. Run predictor
#ifdef IPTCORE_PADDLE_BENCHMARK
    int warmup = 10;
    int repeats = 400;
    Timer timeInstance;
    double first_duration{-1};
    for (size_t widx = 0; widx < warmup; ++widx) {
        if (widx == 0) {
            timeInstance.startTimer();
            infer_specific_item(predictor);
            first_duration = timeInstance.getCostTimer();
        } else {
            infer_specific_item(predictor);
        }
    }

    double sum_duration = 0.0;
    double max_duration = 1e-5;
    double min_duration = 1e5;
    double avg_duration = -1;
    for (size_t ridx = 0; ridx < repeats; ++ridx) {
        timeInstance.startTimer();

        infer_specific_item(predictor);

        double duration = timeInstance.getCostTimer();
        sum_duration += duration;
        max_duration = duration > max_duration ? duration : max_duration;
        min_duration = duration < min_duration ? duration : min_duration;
//        std::cout << "run_idx:" << ridx + 1 << " / " << repeats << ": " << duration
//                  << " ms" << std::endl;
        if (first_duration < 0) {
            first_duration = duration;
        }
    }
    avg_duration = sum_duration / static_cast<float>(repeats);
    std::cout << "\n======= benchmark summary =======\n"
              << "warmup:" << warmup << "\n"
              << "repeats:" << repeats << "\n"
              << "*** time info(ms) ***\n"
              //<< "1st_duration:" << first_duration << "\n"
              << "max_duration:" << max_duration << "\n"
              << "min_duration:" << min_duration << "\n"
              << "avg_duration:" << avg_duration << "\n";
#else
    infer_specific_item(predictor);
#endif

    // 5. Get output
}

int main(int argc, char** argv) {
    UserPersonaInfer user_persona_infer;
#ifdef IPTCORE_PADDLE_MOBILE
//    user_persona_infer.create_paddle_light_predictor(
//        "D:\\baidu\\baiduinput\\inputtools\\paddle_lite\\wenxin\\model_x86.nb");
    user_persona_infer.create_paddle_light_predictor(
        "./model_naive_buffer_arm.nb");
    std::cout << "xiarj" << std::endl;
#else
//    user_persona_infer.create_paddle_full_predictor(
//        "D:\\baidu\\baiduinput\\inputtools\\paddle_lite\\honor_2_11\\cls_ernie_3.0_tiny_fc_ch_dy_15_3L128H_decrypt_inference_1");
#endif
    //user_persona_infer.prepare("D:\\baidu\\baiduinput\\inputtools\\paddle_lite\\wenxin\\xia.txt");
    user_persona_infer.prepare("./xia.txt");
    user_persona_infer.infer();

    return 0;
}

