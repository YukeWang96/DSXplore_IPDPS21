#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// #define gather_forward_cpp

#ifdef gather_forward_cpp || scatter_backward_cpp
std::vector<torch::Tensor> scc_cuda_forward(
    torch::Tensor input,
    torch::Tensor output,
    int input_unit_dim,
    float overlap
);

std::vector<torch::Tensor> scc_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor d_input,
    int input_unit_dim,
    float overlap
);

std::vector<torch::Tensor> scc_forward(
    torch::Tensor input,
    torch::Tensor output,
    int input_unit_dim,
    float overlap
) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  return scc_cuda_forward(input, output, input_unit_dim, overlap);
}

std::vector<torch::Tensor> scc_backward(
    torch::Tensor d_output,
    torch::Tensor d_input,
    int input_unit_dim,
    float overlap) {

  CHECK_INPUT(d_output);
  CHECK_INPUT(d_input);

  return scc_cuda_backward(d_output, d_input, input_unit_dim, overlap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &scc_forward, "RPW forward (CUDA)");
  m.def("backward", &scc_backward, "RPW backward (CUDA)");
}

#else //conventional backpropagation
std::vector<torch::Tensor> scc_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    float overlap
);

std::vector<torch::Tensor> scc_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor d_input,
    torch::Tensor d_weights,
    float overlap
);

std::vector<torch::Tensor> scc_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    float overlap
) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(output);
  return scc_cuda_forward(input, weights, output, overlap);
}

std::vector<torch::Tensor> scc_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor d_input,
    torch::Tensor d_weights,
    float overlap) {

  CHECK_INPUT(d_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(d_input);
  CHECK_INPUT(d_weights);

  return scc_cuda_backward(d_output, input, weights, d_input, d_weights, overlap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &scc_forward, "RPW forward (CUDA)");
  m.def("backward", &scc_backward, "RPW backward (CUDA)");
}
#endif