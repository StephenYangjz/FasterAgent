# Benchmark

## Using previous tools

[Continuous Batching](https://github.com/zinccat/llm-continuous-batching-benchmarks)

0. Go to `API/` and generate dataset using `python dataset_gen.py`
1. `CUDA_VISIBLE_DEVICES=0 ./launch_scripts/launch_vllm --port=8000 --model=ToolBench/ToolLLaMA-2-7b-v2`
2. Use ```bash
   python benchmark_throughput_tool.py --backend='vLLM' --port=8000 --prompts_filename=../API/api_query_data.json --fixed_max_tokens=8192 --verbose
   ```
   instead of
   ```bash
   python benchmark_throughput.py --backend='vLLM' --port=8000 --gen_random_prompts --allow_variable_generation_length --random_prompt_count=10 --random_prompt_lens_mean=30 --random_prompt_lens_range=10  --variable_response_lens_mean=50 --variable_response_lens_range=20 --variable_response_lens_distribution=uniform
   ```
   (for original benchmark code)
   You may need to change parameters