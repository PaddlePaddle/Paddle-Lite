#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "paddle_api.h" // NOLINT
#include <numeric>
#include <string.h>
#include "lite/backends/arm/math/funcs.h"
using namespace paddle::lite_api;
inline float32x4_t log_ps(float32x4_t x) {
  float32x4_t one = vdupq_n_f32(1);

  x = vmaxq_f32(x, vdupq_n_f32(0));  // force flush to zero on denormal values
  uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

  int32x4_t ux = vreinterpretq_s32_f32(x);

  int32x4_t emm0 = vshrq_n_s32(ux, 23);

  // keep only the fractional part
  ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
  ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
  x = vreinterpretq_f32_s32(ux);

  emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
  float32x4_t e = vcvtq_f32_s32(emm0);

  e = vaddq_f32(e, one);

  // part2:
  // if( x < SQRTHF ) {
  //   e -= 1;
  //   x = x + x - 1.0;
  // } else {
  //   x = x - 1.0;
  // }
  //
  uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
  float32x4_t tmp =
      vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
  x = vsubq_f32(x, one);
  e = vsubq_f32(
      e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
  x = vaddq_f32(x, tmp);

  float32x4_t z = vmulq_f32(x, x);

  float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
  y = vmulq_f32(y, x);

  y = vmulq_f32(y, z);

  tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
  y = vaddq_f32(y, tmp);

  tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
  y = vsubq_f32(y, tmp);

  tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
  x = vaddq_f32(x, y);
  x = vaddq_f32(x, tmp);
  x = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(x), invalid_mask));  // negative arg will be NAN
  return x;
}
void act_log(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    float32x4_t exp_vec = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = log_ps(vld1q_f32(ptr_in_thread));
      vst1q_f32(ptr_out_thread, exp_vec);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = logf(ptr_in_thread[0]);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = logf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}
void Predict(std::shared_ptr<PaddlePredictor> &_predictor, const std::vector<std::int64_t> &parsed_input, std::string &output);

int64_t ShapeProduction(const shape_t &shape)
{
    int64_t res = 1;
    for (auto i : shape)
        res *= i;
    return res;
}

std::string trim(const std::string& str) {
    std::string ret;
    int i = 0;
    while (i < str.length() && (str[i] == ' ' || str[i] == '\t')) {
        i++;
    }
    int j = str.length() - 1;
    while (j >= 0 && (str[j] == ' ' || str[j] == '\t')) {
        j--;
    }
    if (i <= j) {
        ret = str.substr(i, j - i + 1);
    } else {
        ret = "";
    }
    return ret;
}


std::int64_t to_int64(std::string input) {
    std::stringstream ss;
    ss << input;
    int x;
    ss >> x;
    std::int64_t m = static_cast<std::int64_t>(x);
	std::cout<<"x:"<<m<<std::endl;
    return m;
}

void string_split(
        const std::string& str,
        std::vector<std::int64_t>& substr_list) {
    substr_list.clear();
    if (str.size() == 0) {
        return;
    }
    if (str.size() > 0) {
        std::string::size_type start_index = 0;
        std::string::size_type cur_index = str.find(" ", start_index);
        while (cur_index != std::string::npos) {
            substr_list.push_back(to_int64(str.substr(start_index, cur_index - start_index)));
            start_index = cur_index + 1;
            cur_index = str.find(" ", start_index);
        }
        substr_list.push_back(to_int64(str.substr(start_index, str.size() - start_index)));
    }
}

void RunModel(std::string model_dir)
{
//	char *s = (char*)malloc(12);
  //  strcpy(s, "Hello world!");
//    printf("string is: %s\n", s);
//    free(s);
    //exit(0);
    //return;
    float indata[2] = {2.2, 3.3};
    float outdata[2] = {0, 0};
    act_log(indata, outdata, 2,1);
#if 0
    CxxConfig cxx_config;
    model_dir = "./models";
    cxx_config.set_model_file(model_dir + "/" + "__model__");
    cxx_config.set_param_file(model_dir + "/" + "__params__");
    cxx_config.set_threads(1);
    std::vector<Place> valid_places{Place{TARGET(kARM), PRECISION(kInt64)}};
    valid_places.insert(valid_places.begin(), Place{TARGET(kARM), PRECISION(kInt32)});
    valid_places.insert(valid_places.begin(), Place{TARGET(kARM), PRECISION(kFloat)});
    valid_places.insert(valid_places.begin(), Place{TARGET(kARM), PRECISION(kInt64)});
    cxx_config.set_valid_places(valid_places);
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(cxx_config);
 #else
    model_dir = "/home/firefly/myq/trans/trans.nb";
  MobileConfig config;
    config.set_model_from_file(model_dir);
      std::shared_ptr<PaddlePredictor> predictor =
            CreatePaddlePredictor<MobileConfig>(config);
#endif
    std::string output;
    std::string input;
#if 1    
    //std::vector<std::int64_t> parsed_input = {8098, 963, 7089, 4601, 6620, 4945, 5305, 9694, 7223, 6761, 1};
    std::vector<std::int64_t> parsed_input = {93, 7453, 10430, 4693, 6307, 4284, 4693, 7262, 1};
    std::cout << "passss:size:"<<parsed_input.size()<< std::endl;
    Predict(predictor, parsed_input, output);
    std::cout << output << std::endl;
#else
   while (std::getline(std::cin, input))
    {
         //input = trim("8098, 963, 7089, 4601, 6620, 4945, 5305, 9694, 7223, 6761, 1");//trim(input);
         input = trim(input);
         std::cout<<"input:"<<input<<std::endl;
         std::vector<std::int64_t> parsed_input;
         string_split(input, parsed_input);

         std::cout << input << std::endl;
         for (int i = 0; i < parsed_input.size(); ++i) {
             if (i != parsed_input.size() - 1) {
                 std::cout << parsed_input[i] << " ";
             } else {
                 std::cout << parsed_input[i] << std::endl;
             }
         }
    std::cout << "passss:size:"<<parsed_input.size()<< std::endl;
         Predict(predictor, parsed_input, output);
         std::cout << output << std::endl;
    }
#endif
}

void Predict(std::shared_ptr<PaddlePredictor> &_predictor,  const std::vector<std::int64_t> &parsed_input, std::string &output)
{
    // 构造数据
    // query:   我来自中国,北京.
    // subword: 我 来自 中国 _ ,_ 北京 _ ._
    // ids: 93 7453 10430 4693 6307 4284 4693 7262 1
    // batch_size=1, seq_len=9
    // std::vector<std::int64_t> parsed_input = {93, 7453, 10430, 4693, 6307, 4284, 4693, 7262, 1};
    // std::vector<std::int64_t> parsed_input = {8098, 963, 7089, 4601, 6620, 4945, 5305, 9694, 7223, 6761, 1};
    int batch_size = 1, max_seq_len = parsed_input.size();
    std::vector<int64_t> seq_lens = {parsed_input.size()};
    int64_t eos_id = 1, _n_head = 8;

    // 第1个输入: src_word [(batch_size, seq_len), "int64", 2]
    std::unique_ptr<Tensor> src_word(std::move(_predictor->GetInput(0)));
    src_word->Resize({batch_size, max_seq_len});
    //std::vector<std::vector<size_t>> src_word_lod = {};
    //src_word->SetLoD(src_word_lod);
    auto src_word_data_ptr = src_word->mutable_data<int64_t>();
    // std::cout<<"------------------0"<<std::endl;
    for (int i = 0; i < parsed_input.size(); ++i)
    {
        src_word_data_ptr[i] = parsed_input[i];
        // std::cerr << parsed_input[i] << std::endl;
    }

    // 第2个输入: src_pos [(batch_size, seq_len), "int64"]
    std::unique_ptr<Tensor> src_pos(std::move(_predictor->GetInput(1)));
    src_pos->Resize({batch_size, max_seq_len});
    std::vector<int64_t> src_pos_data(batch_size * max_seq_len, 0);
    for (int i = 0; i < batch_size; ++i)
    {
        std::iota(src_pos_data.begin() + i * max_seq_len, src_pos_data.begin() + i * max_seq_len + seq_lens[i], 0);
    }
    auto src_pos_data_ptr = src_pos->mutable_data<int64_t>();
    std::cout<<"------------------1"<<std::endl;
    for (int i = 0; i < src_pos_data.size(); ++i)
    {
        src_pos_data_ptr[i] = src_pos_data[i];
        std::cerr << src_pos_data_ptr[i] << std::endl;
    }

    // 第3个输入: src_slf_attn_bias -> [(batch_size, n_head, seq_len, seq_len), "float32"]
    std::unique_ptr<Tensor> src_slf_attn_bias(std::move(_predictor->GetInput(2)));
    src_slf_attn_bias->Resize({batch_size, _n_head, max_seq_len, max_seq_len});
    //std::vector<std::vector<size_t>> src_bias_lod = {};
    //src_slf_attn_bias->SetLoD(src_bias_lod);
    std::vector<float> src_slf_attn_bias_data;
    if (batch_size == 1)
    {
        src_slf_attn_bias_data.resize(batch_size * _n_head * max_seq_len * max_seq_len, 0);
    }
    else
    {
        exit(1);
    }
    auto src_slf_attn_bias_data_ptr = src_slf_attn_bias->mutable_data<float>();
    // std::cout<<"------------------2"<<std::endl;
    for (int i = 0; i < ShapeProduction(src_slf_attn_bias->shape()); ++i)
    {
        src_slf_attn_bias_data_ptr[i] = 0.f;
    }

    // 第4个输入: trg_word ->  [(batch_size, seq_len), "int64", 2]
    std::unique_ptr<Tensor> trg_word(std::move(_predictor->GetInput(3)));
    trg_word->Resize({batch_size, 1});
    std::vector<size_t> trg_word_lod(0, 1);
    trg_word->SetLoD({{0, 1}, {0, 1}});

    auto trg_word_data_ptr = trg_word->mutable_data<int64_t>();
    // std::cout<<"------------------3"<<std::endl;
    for (int i = 0; i < ShapeProduction(trg_word->shape()); ++i)
    {
        trg_word_data_ptr[i] = 0;
    }

    // 第5个输入: init_score -> [(batch_size, 1), "float32", 2]
    std::unique_ptr<Tensor> init_score(std::move(_predictor->GetInput(4)));
    init_score->Resize({1, 1});
    init_score->SetLoD({{0, 1}, {0, 1}});
    // std::cout<<"------------------4"<<std::endl;
    // std::cout<<"------------------4 num:"<<batch_size *1;
    auto init_score_data_ptr = init_score->mutable_data<float>();
    for (int i = 0; i < ShapeProduction(init_score->shape()); ++i)
    {
        init_score_data_ptr[i] = 0;
    }

    // 第6个输入: init_idx -> [(batch_size, ), "int32"]
    std::unique_ptr<Tensor> init_idx(std::move(_predictor->GetInput(5)));
    init_idx->Resize({batch_size});
    auto init_idx_data_ptr = init_idx->mutable_data<int>();
    // std::cout<<"------------------5"<<std::endl;
    for (int i = 0; i < ShapeProduction(init_idx->shape()); ++i)
    {
        init_idx_data_ptr[i] = 0;
    }

    // 第7个输入: trg_src_attn_bias -> [(batch_size, n_head, seq_len, seq_len), "float32"]
    std::unique_ptr<Tensor> trg_src_attn_bias(std::move(_predictor->GetInput(6)));
    trg_src_attn_bias->Resize({batch_size, _n_head, 1, max_seq_len});
    auto trg_src_attn_bias_data_ptr = trg_src_attn_bias->mutable_data<float>();
    for (int i = 0; i < ShapeProduction(trg_src_attn_bias->shape()); ++i)
    {
        trg_src_attn_bias_data_ptr[i] = 332;
    }

    // std::cout<<"-----------------------begin"<<std::endl;
    _predictor->Run();
    // std::cout<<"-----------------------end"<<std::endl;
    // 解析翻译结果
    std::unique_ptr<const Tensor> seq_ids(std::move(_predictor->GetOutput(0)));
    auto out_lod = seq_ids->lod();
    // std::cerr << "out_lod[0]=" << std::endl;
    // for (size_t i = 0; i < out_lod[0].size(); ++i) {
    //     std::cerr << out_lod[0][i] << " ";
    // }
    // std::cerr << std::endl;

    // std::cerr << "out_lod[1]=" << std::endl;
    // for (size_t i = 0; i < out_lod[1].size(); ++i) {
    //     std::cerr << out_lod[1][i] << " ";
    // }
    // std::cerr << std::endl;

    // lod [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
    // lod[0]: there are 2 source sentences, beam width is 3.
    // lod[1]: the first source sentence has 3 hyps; the lengths are 12, 12, 16
    //         the second source sentence has 3 hyps; the lengths are 14, 13, 15

    // 取所有翻译结果
    std::vector<std::vector<std::vector<int64_t>>> beam_search_res;

    // 遍历每个源语句
    for (size_t i = 0; i < out_lod[0].size() - 1; ++i)
    {
        size_t start = out_lod[0][i];   // 0
        size_t end = out_lod[0][i + 1]; // 3
        std::vector<std::vector<int64_t>> hyps;
        // 遍历该源语句的所有结果
        for (size_t j = 0; j < end - start; ++j)
        {                                               // j < 3
            size_t sub_start = out_lod[1][start + j];   // 翻译结果的起始位置 0
            size_t sub_end = out_lod[1][start + j + 1]; // 翻译结果的结束位置 12
            auto data = seq_ids->mutable_data<int64_t>();
            std::vector<int64_t> hyp;
            for (size_t k = sub_start; k < sub_end && data[k] != eos_id; k++)
            {
                hyp.push_back(data[k]);
            }
            hyps.push_back(std::move(hyp));
        }
        beam_search_res.push_back(std::move(hyps));
    }

    std::vector<std::string> trans_output;
    std::string cur_word;
    // 取 top 1
    for (size_t src_id = 0; src_id < beam_search_res.size(); ++src_id)
    {
        std::string output_str;
        if (beam_search_res[src_id].size() > 0)
        {
            // 0 表示取第1个结果
            for (size_t trg_id = 0; trg_id < beam_search_res[src_id][0].size(); ++trg_id)
            {
                cur_word = std::to_string(beam_search_res[src_id][0][trg_id]);
                output_str = output_str + cur_word + " ";
            }
        }
        trans_output.push_back(output_str);
    }

    // for (size_t i = 0; i < trans_output.size(); ++i)
    // {
    //     std::cout << trans_output[i] << std::endl;
    // }

    output = trans_output[0];

    // 输出翻译结果
    // auto seq_ids_data = seq_ids->data<int64_t>();
    // for (size_t i = 0; i < ShapeProduction(seq_ids->shape()); ++i) {
    //     std::cerr << seq_ids_data[i] << " ";
    // }
}

int main(int argc, char **argv)
{
#if 0
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " [config]" << std::endl;
        exit(1);
    }
    RunModel(argv[1]);
#else
    RunModel(argv[1]);
#endif
    return 0;
}
