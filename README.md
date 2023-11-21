# FasterAgent

[Benchmark](benchmark.md)
[Profiling](./profiling/)
[Design Doc](https://docs.google.com/document/d/1XXtushjemOyqUHXWH3OEYrdWspEzNrU7cgo2yCECx4Q/edit?usp=sharing)

## Usage
vLLM with tool use:
```shell
cd vllm
# server
python -m vllm.entrypoints.api_server --model /projects/bcby/qlao/ToolLLaMA-2-7b-v2
# client
python /u/qlao/repos/FasterAgent/vllm/examples/api_client_tool_use.py
```