/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// Utility function for attention softmax.
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// new

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE = 0> // Zero means no partitioning.
__device__ void paged_attention_kernel_new(
  float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
  // Todo: recheck q size
  const scalar_t* __restrict__ q,         // [num_tokens, num_heads, prompt_tokens, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int* __restrict__ head_mapping,   // [num_heads]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
//   const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  const int prompt_tokens,
  ){
    const int token_idx = blockIdx.y; // Todo: fix this
    const int partition_idx = blockIdx.z;
    const int max_num_partitions = gridDim.z;
    constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
    const int context_len = prompt_tokens; //context_lens[seq_idx];
    if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
        // No work to do. Terminate the thread block.
        return;
    }

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

    // [start_block_idx, end_block_idx) is the range of blocks to process.
    const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
    const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
    const int num_blocks = end_block_idx - start_block_idx;

    // [start_token_idx, end_token_idx) is the range of tokens to process.
    const int start_token_idx = start_block_idx * BLOCK_SIZE;
    const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
    const int num_tokens = end_token_idx - start_token_idx;

    constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
    constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
    assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
    constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane = thread_idx % WARP_SIZE;

    const int head_idx = blockIdx.x;
    const int num_heads = gridDim.x;
    const int kv_head_idx = head_mapping[head_idx];
    const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

    // A vector type to store a part of a key or a query.
    // The vector size is configured in such a way that the threads in a thread group
    // fetch or compute 16 bytes at a time.
    // For example, if the size of a thread group is 4 and the data type is half,
    // then the vector size is 16 / (4 * sizeof(half)) == 2.
    constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
    using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
    using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

    constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE * prompt_tokens / THREAD_GROUP_SIZE;
    constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

    const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
    const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

    // Load the query to registers.
    // Each thread in a thread group has a different part of the query.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
    // th vectors of the query, and so on.
    // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
    const scalar_t* q_ptr = q + head_idx * HEAD_SIZE;
    __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
        const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
        const int token_idx = i * VEC_SIZE / 
        q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    }

    // q, [num_seqs, num_heads, prompt_tokens, head_size]


  }
}

// TODO(woosuk): Tune NUM_THREADS.
template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void paged_attention_v3_launcher(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  int prompt_tokens = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* head_mapping_ptr = reinterpret_cast<int*>(head_mapping.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V3(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V3(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V3(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V3(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V3(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V3(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V3_LAUNCHER(T, BLOCK_SIZE)                             \
  paged_attention_v3_launcher<T, BLOCK_SIZE>(                       \
    out,                                                            \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    head_mapping,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len,                                                \
    alibi_slopes);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V3_LAUNCHER_BLOCK_SIZE(T)                              \
  switch (block_size) {                                             \
    case 8:                                                         \
      CALL_V3_LAUNCHER(T, 8);                                       \
      break;                                                        \
    case 16:                                                        \
      CALL_V3_LAUNCHER(T, 16);                                      \
      break;                                                        \
    case 32:                                                        \
      CALL_V3_LAUNCHER(T, 32);                                      \
      break;                                                        \
    default:                                                        \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);   \
      break;                                                        \
  }