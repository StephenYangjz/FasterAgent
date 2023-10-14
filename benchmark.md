# Benchmark

## Using previous tools

[Continuous Batching](https://github.com/zinccat/llm-continuous-batching-benchmarks)

1. `CUDA_VISIBLE_DEVICES=0 ./launch_scripts/launch_vllm --port=8000`
2. ```bash
   python benchmark_throughput.py --backend='vLLM' --port=8000 --gen_random_prompts --allow_variable_generation_length --random_prompt_count=10 --random_prompt_lens_mean=30 --random_prompt_lens_range=10  --variable_response_lens_mean=50 --variable_response_lens_range=20 --variable_response_lens_distribution=uniform
   ```
   You may need to change parameters
