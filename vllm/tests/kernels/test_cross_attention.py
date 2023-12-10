import random
from typing import Tuple

import pytest
import torch

from vllm import attention_ops
from vllm.utils import get_max_shared_memory_bytes

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
LLAMA_MAX_SEQ_LEN = 8192
NUM_BLOCKS = 10000  # Arbitrary values for testing

# DTYPES = [torch.half, torch.bfloat16, torch.float]
DTYPES = [torch.float]
NUM_GEN_SEQS = [2]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
LLAMA_NUM_HEADS = [(32, 32)]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
LLAMA_HEAD_SIZES = [128]
BLOCK_SIZES = [16, 32]
LLAMA_BLOCK_SIZES = [16]
SEEDS = [0]
SEQ_LENS = [256]


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", LLAMA_NUM_HEADS)
@pytest.mark.parametrize("head_size", LLAMA_HEAD_SIZES)
@pytest.mark.parametrize("block_size", LLAMA_BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_cross_paged_attention(
    kv_cache_factory,
    num_seqs: int,
    seq_lens: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    max_seq_len = LLAMA_MAX_SEQ_LEN - seq_lens + 1

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        seq_lens,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None

    context_lens = []
    for i in range(num_seqs):
        if i == num_seqs - 1:
            context_len = max_seq_len
        else:
            context_len = random.randint(1, max_seq_len)
        context_lens.append([context_len + j for j in range(seq_lens)])

    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")
    max_context_len = torch.max(context_lens).item()

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    ref_output = torch.empty_like(query)
    tmp_output = torch.empty_like(query[:, 0])
    start_paged = torch.cuda.Event(enable_timing=True)
    end_paged = torch.cuda.Event(enable_timing=True)
    start_paged.record()
    for q_token_idx in range(seq_lens):
        attention_ops.paged_attention_v1(
            tmp_output,
            query[:, q_token_idx].contiguous(),
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens[:, q_token_idx].contiguous().unsqueeze(-1),
            block_size,
            max_context_len - seq_lens + 1 + q_token_idx,
            alibi_slopes,
        )
        ref_output[:, q_token_idx] = tmp_output
    end_paged.record()

    # Call the paged cross attention kernel.
    output = torch.empty_like(query)
    start_cross = torch.cuda.Event(enable_timing=True)
    end_cross = torch.cuda.Event(enable_timing=True)
    start_cross.record()
    attention_ops.paged_cross_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )
    end_cross.record()
    torch.cuda.synchronize()
    print(f"{start_paged.elapsed_time(end_paged)}ms")
    print(f"{start_cross.elapsed_time(end_cross)}ms")

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=0, rtol=0)
