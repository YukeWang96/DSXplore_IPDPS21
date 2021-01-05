#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <vector>

// #define regular_backward // one thread per output point
#define input_centric_backward
// #define one_thread_per_dimension_backward

template <typename scalar_t>
__global__ void scc_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> new_tensor,
    int batch_size,
    int input_channel,
    int input_height,
    int input_width,
    int output_channel,
    int input_unit_dim,
    float overlap
);

template <typename scalar_t>
__global__ void scc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_weights,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    float overlap
);
////////////////////////////////////////////
// foward pass
////////////////////////////////////////////
std::vector<torch::Tensor> scc_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor new_tensor,
    float overlap
) {

    // input: batch_size * input_channel * input_width * input_height.
    const int batch_size = input.size(0);
    const int input_channel = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // output: batch_size * opt_channel * input_width * input_height.
    const int output_channel = weights.size(0);

    // weight: output_channel * input_units_dim * 1.
    const int input_unit_dim = weights.size(1); 

    // new tensor for output.
    const int threads = 1024;
    const int blocks = (batch_size * output_channel * input_width * input_height + threads - 1) / threads; 

    AT_DISPATCH_FLOATING_TYPES(input.type(), "scc_forward_cuda", ([&] {
                                scc_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                    input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    new_tensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    batch_size,
                                    input_channel,
                                    input_height,
                                    input_width,
                                    output_channel,
                                    input_unit_dim,
                                    overlap
                                );
                            }));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {new_tensor};
}

template <typename scalar_t>
__global__ void scc_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> new_tensor,
    int batch_size,
    int input_channel,
    int input_height,
    int input_width,
    int output_channel,
    int input_unit_dim,
    float overlap
) {
  const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int g_dim = batch_size * output_channel * input_width * input_height;
  const int item_size_dim = output_channel * input_height * input_width;
  const int feature_map_dim = input_height * input_width;

  const int item_idx = g_idx / item_size_dim;
  const int item_channel_idx =  (g_idx - item_idx * item_size_dim) / feature_map_dim;
  const int item_feat_y_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) / input_width;
  const int item_feat_x_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) % input_width;
  const int b = item_idx;
  const int c = item_channel_idx;
  const int y = item_feat_y_idx;
  const int x = item_feat_x_idx;
  
  const int input_c_start = __float2int_rd(item_channel_idx * (1 - overlap) * input_unit_dim) % input_channel;
  const int input_c_end = (input_c_start + input_unit_dim) % input_channel;
  const int input_x = x;
  const int input_y = y;

//   new_tensor[b][c][y][x] = 5;
//   printf("gid: %d, total thread: %d\n", g_idx, g_dim);
  if (g_idx < g_dim) {
        float tmp = 0;
        // printf("input_c_start, %d, input_c_end, %d\n", input_c_start, input_c_end);
        if (input_c_start < input_c_end)
            for(int c_input_d = input_c_start; c_input_d < input_c_end; c_input_d++){
                tmp += input[b][c_input_d][input_y][input_x] * weights[c][c_input_d - input_c_start];
            }
        else
        {
            for(int c_input_d = input_c_start; c_input_d < input_channel; c_input_d++){
                tmp += input[b][c_input_d][input_y][input_x] * weights[c][c_input_d - input_c_start];
            }
            for(int c_input_d = 0; c_input_d < input_c_end; c_input_d++){
                tmp += input[b][c_input_d][input_y][input_x] * weights[c][c_input_d + input_channel - input_c_start];
            } 
        }
        new_tensor[b][c][y][x] = tmp;
        // printf("gid: %d, new tensor (%d, %d, %d, %d) --- %f\n", g_idx, b, c, y, x, new_tensor[0][0][0][0]);
  }
}

////////////////////////////////////////////
// backward pass
////////////////////////////////////////////
#ifdef regular_backward
std::vector<torch::Tensor> scc_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor d_input,
    torch::Tensor d_weights,
    float overlap
) {

    // input: batch_size * input_channel * input_width * input_height.
    const int batch_size = d_output.size(0);
    const int output_channel = d_output.size(1);
    const int height = d_output.size(2);
    const int width = d_output.size(3);

    // output: batch_size * opt_channel * input_width * input_height.
    const int input_channel = d_input.size(1);

    // weight: output_channel * input_units_dim * 1.
    const int input_unit_dim = weights.size(1); 

    const int threads = 1024;
    const int blocks = (batch_size * output_channel * width * height + threads - 1) / threads; 

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "scc_backward_cuda", ([&] {
                            scc_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                    d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    d_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    batch_size,
                                    input_channel,
                                    height,
                                    width,
                                    output_channel,
                                    input_unit_dim,
                                    overlap
                                );
                            }));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {d_input, d_weights};
}

template <typename scalar_t>
__global__ void scc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_weights,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    float overlap
) {
  const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int g_dim = batch_size * output_channel * width * height;
  const int item_size_dim = output_channel * height * width;
  const int feature_map_dim = height * width;

  const int item_idx = g_idx / item_size_dim;
  const int item_channel_idx =  (g_idx - item_idx * item_size_dim) / feature_map_dim;
  const int item_feat_y_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) / width;
  const int item_feat_x_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) % width;
  const int b = item_idx;
  const int c = item_channel_idx;
  const int y = item_feat_y_idx;
  const int x = item_feat_x_idx;
  
  const int input_c_start = __float2int_rd(item_channel_idx * (1 - overlap) * input_unit_dim) % input_channel;
  const int input_c_end = (input_c_start + input_unit_dim) % input_channel;
  const int input_x = x;
  const int input_y = y;

//   printf("gid: %d, total thread: %d\n", g_idx, g_dim);
  if (g_idx < g_dim) {
        // printf("input_c_start, %d, input_c_end, %d\n", input_c_start, input_c_end);
        float d_tmp = d_output[b][c][y][x];
        if (input_c_start < input_c_end)
            for(int c_input_d = input_c_start; c_input_d < input_c_end; c_input_d++){
                atomicAdd((float*)&d_input[b][c_input_d][input_y][input_x], weights[c][c_input_d - input_c_start] * d_tmp);
                // d_input[b][c_input_d][input_y][input_x] += weights[c][c_input_d - input_c_start] * d_tmp);
                atomicAdd((float*)&d_weights[c][c_input_d - input_c_start], input[b][c_input_d][input_y][input_x] * d_tmp);
            }
        else
        {
            for(int c_input_d = input_c_start; c_input_d < input_channel; c_input_d++){
                // d_input[b][c_input_d][input_y][input_x] += weights[c][c_input_d - input_c_start] * d_tmp;
                atomicAdd((float*)&d_input[b][c_input_d][input_y][input_x], weights[c][c_input_d - input_c_start] * d_tmp);
                atomicAdd((float*)&d_weights[c][c_input_d - input_c_start], input[b][c_input_d][input_y][input_x] * d_tmp);
            }
            for(int c_input_d = 0; c_input_d < input_c_end; c_input_d++){
                // d_input[b][c_input_d][input_y][input_x] += weights[c][c_input_d + input_channel - input_c_start] * d_tmp;
                atomicAdd((float*)&d_input[b][c_input_d][input_y][input_x], weights[c][c_input_d + input_channel - input_c_start] * d_tmp);
                atomicAdd((float*)&d_weights[c][c_input_d + input_channel - input_c_start], input[b][c_input_d][input_y][input_x] * d_tmp);
            } 
        }
        // printf("gid: %d, new tensor (%d, %d, %d, %d)\n", g_idx, b, c, y, x);
  }
}
#endif

////////////////////////////////////////////
// backward pass
////////////////////////////////////////////
#ifdef scatter_backward
std::vector<torch::Tensor> scc_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor d_input,
    int input_unit_dim,
    float overlap
) {

    // input: batch_size * input_channel * input_width * input_height.
    const int batch_size = d_output.size(0);
    const int output_channel = d_output.size(1);
    const int height = d_output.size(2);
    const int width = d_output.size(3);
    const int input_channel = d_input.size(1);

    const int threads = 1024;
    const int blocks = (batch_size * output_channel * width * height + threads - 1) / threads; 

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "scc_backward_cuda", ([&] {
                            scc_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                    d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    d_input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    batch_size,
                                    input_channel,
                                    height,
                                    width,
                                    output_channel,
                                    input_unit_dim,
                                    overlap
                                );
                            }));
    return {d_input};
}

template <typename scalar_t>
__global__ void scc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    float overlap
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int g_dim = batch_size * output_channel * input_unit_dim * height * width;
    
  if (g_idx < g_dim) {
   
    const int feature_map_dim = height * width;
    const int per_channel_dim = input_unit_dim * feature_map_dim;
    const int item_size_dim = output_channel * input_unit_dim * feature_map_dim;

    const int item_idx = g_idx / item_size_dim;
    const int item_channel_idx = (g_idx - item_idx * item_size_dim) / per_channel_dim;
    const int input_unit_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * per_channel_dim) / feature_map_dim;
  
    const int feature_map_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * per_channel_dim - input_unit_idx * feature_map_dim);
    const int item_feat_y_idx = feature_map_idx / width;
    const int item_feat_x_idx = feature_map_idx % width;
  
    const int b = item_idx;
    const int c = item_channel_idx;
    const int y = item_feat_y_idx;
    const int x = item_feat_x_idx;
    
    const int input_c_start = __float2int_rd(item_channel_idx * (1 - overlap) * input_unit_dim) % input_channel;
  
    if (g_idx < g_dim) {
        atomicAdd((float*)&(d_input[b][(input_c_start + input_unit_idx) % input_channel][y][x]), d_output[b][c * input_unit_dim + input_unit_idx][y][x]);
    }
  }
}
#endif

#ifdef input_centric_backward
std::vector<torch::Tensor> scc_cuda_backward(
        torch::Tensor d_output,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor d_input,
        torch::Tensor d_weights,
        float overlap
    ) {
    
        // input: batch_size * input_channel * input_width * input_height.
        const int batch_size = d_output.size(0);
        const int output_channel = d_output.size(1);
        const int height = d_output.size(2);
        const int width = d_output.size(3);
    
        // output: batch_size * opt_channel * input_width * input_height.
        const int input_channel = d_input.size(1);
    
        // weight: output_channel * input_units_dim * 1.
        const int input_unit_dim = weights.size(1); 
    
        const int threads = 1024;
        const int blocks = (batch_size * input_channel * width * height + threads - 1) / threads; 
    
        AT_DISPATCH_FLOATING_TYPES(d_output.type(), "scc_backward_cuda", ([&] {
                                scc_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                        d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                        d_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        batch_size,
                                        input_channel,
                                        height,
                                        width,
                                        output_channel,
                                        input_unit_dim,
                                        overlap
                                    );
                                }));
        // check for error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        return {d_input, d_weights};
    }
    
template <typename scalar_t>
__global__ void scc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_weights,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    float overlap
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int g_dim = batch_size * input_channel * width * height;

//   printf("gid: %d, total thread: %d\n", g_idx, g_dim);
    if (g_idx < g_dim) {
        const int item_size_dim = input_channel * height * width;
        const int feature_map_dim = height * width;
        const int item_idx = g_idx / item_size_dim;
        const int item_channel_idx =  (g_idx - item_idx * item_size_dim) / feature_map_dim;
        const int b = item_idx;
        const int y = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) / width;
        const int x = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) % width;
        int cid;
        int const_term = __float2int_rd((1-overlap) * input_unit_dim);

        for (int v_cid = item_channel_idx; true; v_cid += input_channel){
            
            #ifdef debug
            if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3)
            {
                printf("v_cid: %d\n", v_cid);
            }
            #endif

            int output_start_idx = v_cid / const_term;
            int output_start_offset = v_cid % const_term;

            int output_end_idx = 0;
            int output_end_offset = 0;
            
            if (v_cid < input_unit_dim){
                output_end_idx = 0;
                output_end_offset = v_cid;
            } 
            else{
                output_end_idx = __float2int_rd((v_cid - input_unit_dim) * 1.0f / const_term) + 1;
                output_end_offset = v_cid - const_term * output_end_idx; // (__float2int_rd((1 - overlap) * (output_end_idx - 1)) + 1) * input_unit_dim;
            }

            #ifdef debug
            if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3)
            printf("opt_start_idx: %d\nopt_start_offset: %d\nopt_end_idx: %d\nopt_end_offset: %d\n\n", output_start_idx, output_start_offset, output_end_idx, output_end_offset);
            #endif

            if (output_start_idx >= output_channel && output_end_idx >= output_channel) break;

            cid = v_cid % input_channel;
            if (output_start_idx == output_end_idx){
                d_input[b][cid][y][x] += weights[output_start_idx][output_start_offset] * d_output[b][output_start_idx][y][x];

                #ifdef enforce_atomic
                atomicAdd((float*)&d_weights[output_start_idx][output_start_offset], input[b][cid][y][x] * d_output[b][output_start_idx][y][x]);
                #else
                d_weights[output_start_idx][output_start_offset] += input[b][cid][y][x] * d_output[b][output_start_idx][y][x];
                #endif
            }
            else{
                if (output_start_idx < output_channel)
                {
                    // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 2) printf("1111-output_start_idx < output_channel add\n");
                    d_input[b][cid][y][x] += weights[output_start_idx][output_start_offset] * d_output[b][output_start_idx][y][x];
                    // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3) printf("%f, %f\n", weights[output_start_idx][output_start_offset], d_output[output_start_idx][output_start_offset][y][x]);
                    // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3) printf("d_input[b][cid][y][x], %f\n", d_input[b][cid][y][x]);
                    #ifdef enforce_atomic
                    atomicAdd((float*)&d_weights[output_start_idx][output_start_offset], input[b][cid][y][x] * d_output[b][output_start_idx][y][x]);
                    #else
                    d_weights[output_start_idx][output_start_offset] += input[b][cid][y][x] * d_output[b][output_start_idx][y][x];
                    #endif 
                }
                if (output_end_idx < output_channel)
                {
                    // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 2) printf("2222-output_end_idx < output_channel add\n");
                    d_input[b][cid][y][x] += weights[output_end_idx][output_end_offset] * d_output[b][output_end_idx][y][x];
                    // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3) printf("%f, %f\n", weights[output_end_idx][output_end_offset], d_output[output_end_idx][output_end_offset][y][x]);
                    // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3) printf("d_input[b][cid][y][x], %f\n", d_input[b][cid][y][x]);

                    #ifdef enforce_atomic
                    atomicAdd((float*)&d_weights[output_end_idx][output_end_offset], input[b][cid][y][x] * d_output[b][output_end_idx][y][x]);
                    #else
                    d_weights[output_end_idx][output_end_offset] += input[b][cid][y][x] * d_output[b][output_end_idx][y][x];
                    #endif
                }
                // if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 2) printf("d_input[b][cid][y][x], %f\n", d_input[b][cid][y][x]);
            }

        }
    }
}
#endif


/////////////////////////////////////////////////////////////////////////////
// one thread one dimension of a weight
#ifdef one_thread_per_dimension_backward
std::vector<torch::Tensor> scc_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor d_input,
    torch::Tensor d_weights,
    float overlap
) {

    // input: batch_size * input_channel * input_width * input_height.
    const int batch_size = d_output.size(0);
    const int output_channel = d_output.size(1);
    const int height = d_output.size(2);
    const int width = d_output.size(3);

    // output: batch_size * opt_channel * input_width * input_height.
    const int input_channel = d_input.size(1);

    // weight: output_channel * input_units_dim * 1.
    const int input_unit_dim = weights.size(1); 

    const int threads = 1024;
    const int blocks = ((batch_size * output_channel * width * height) * input_unit_dim + threads - 1) / threads; 

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "scc_backward_cuda", ([&] {
                            scc_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                    d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    d_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    batch_size,
                                    input_channel,
                                    height,
                                    width,
                                    output_channel,
                                    input_unit_dim,
                                    overlap
                                );
                            }));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    return {d_input, d_weights};
}

template <typename scalar_t>
__global__ void scc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_weights,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    float overlap
) {
    const int g_idx = (blockIdx.x * blockDim.x + threadIdx.x) / input_unit_dim;
    const int g_dim = batch_size * output_channel * width * height;

    if (g_idx < g_dim) {
        const int item_size_dim = output_channel * height * width;
        const int feature_map_dim = height * width;
      
        const int item_idx = g_idx / item_size_dim;
        const int item_channel_idx =  (g_idx - item_idx * item_size_dim) / feature_map_dim;
        const int item_feat_y_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) / width;
        const int item_feat_x_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) % width;
        const int b = item_idx;
        const int c = item_channel_idx;
        const int y = item_feat_y_idx;
        const int x = item_feat_x_idx;

        // printf("gid: %d, new tensor (%d, %d, %d, %d)\n", g_idx, b, c, y, x);
        const int input_c_start = __float2int_rd(item_channel_idx * (1 - overlap) * input_unit_dim) % input_channel;
        const int offset = (blockIdx.x * blockDim.x + threadIdx.x) % input_unit_dim;
        const int c_input_d = (input_c_start + offset) % input_channel;
        const int input_x = x;
        const int input_y = y;

        float d_tmp = d_output[b][c][y][x];
        #ifdef enforce_atomic
        atomicAdd((float*)&d_input[b][c_input_d][input_y][input_x], weights[c][offset] * d_tmp);
        atomicAdd((float*)&d_weights[c][offset], input[b][c_input_d][input_y][input_x] * d_tmp);
        #else
        d_input[b][c_input_d][input_y][input_x] += weights[c][offset] * d_tmp;
        d_weights[c][offset] += input[b][c_input_d][input_y][input_x] * d_tmp;
        #endif
    }
}
#endif