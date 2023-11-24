# Benchmark

## Using previous tools

[Continuous Batching](https://github.com/zinccat/llm-continuous-batching-benchmarks)

0. Go to `API/` and generate dataset using `python dataset_gen.py`
1. Change ToolLlama's config (config.json) in `~/.cache/huggingface/hub/models--ToolBench--ToolLLaMA-2-7b-v2/snapshots`, change `max_position_embeddings` to 8192.
2. `CUDA_VISIBLE_DEVICES=0 ./launch_scripts/launch_vllm --port=8000 --model=/scratch/bcby/agent/ToolLLaMA-2-7b-v2 --preemption-mode=SWAP --swap-space=25`
3. Use ```bash
   python benchmark_throughput_tool.py --backend='vLLM' --port=8000 --prompts_filename=../API/api_query_data.json --fixed_max_tokens=8192 --log_latencies --verbose
   ```
   instead of
   ```bash
   python benchmark_throughput.py --backend='vLLM' --port=8000 --gen_random_prompts --allow_variable_generation_length --random_prompt_count=10 --random_prompt_lens_mean=30 --random_prompt_lens_range=10  --variable_response_lens_mean=50 --variable_response_lens_range=20 --variable_response_lens_distribution=uniform
   ```
   (for original benchmark code)
   You may need to change parameters