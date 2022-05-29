#include "cgotorch/jit.h"
#include <vector>

JitModule JitLoad(const char *path, Tensor *tensor) {
    try {
        std::string spath(path);
        torch::jit::script::Module* module = new torch::jit::script::Module();
        *module = torch::jit::load(spath);
        return (void*)module;
    } catch (const std::exception &e) {
         std::cerr << exception_str(e.what());
    }
}


const char *JitForward(JitModule module, Tensor input, Tensor *result) {
  try {
    std::vector<torch::jit::IValue> input_vec;
    input_vec.push_back((torch::Tensor*)input);
    auto out = ((torch::jit::script::Module*)module)->forward(input_vec).toTensor();
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    std::cerr << exception_str(e.what());
  }
}


