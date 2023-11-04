# Profiling

This folder is for profiling of common LLMs.

Usage:

1. First install `llama` from local: `pip install -e .`
2. Follow instruction in Code part.

Findings (all experiments are conducted on a single RTX 3090 GPU):
1. KV cache size: 32 (layers) * [1, 512, 32, 128] (max_batch_size, max_seq_len, n_local_kv_heads, head_dim) * 2 bytes (half) * 2 (k,v) = 256Mb, check `model.py` line 285-286
2. Moving speed: for 256Mb splitted KV caches, moving from GPU/CPU to CPU/GPU each time takes 21ms (a bit slower than the bandwidth of 25Gb/s, maybe caused by separate tensors).
3. Generation speed: Tokens generated: 432, Time taken: 9.19 seconds, Tokens per second: 47.03

|                          | Llama-2-7b-chat  | Llama-2-7b-chat | Llama-2-13b-chat |
| :----------------------- | :--------------  | :-------------  | :--------------- |
| Device                   | RTX 3090x1       | A100x1          | A100x2           |
| KV Cache Size            | 256 MB           | 256 MB          | 200 MB           |
| Moving Time (htod, dtoh) | 21, 21 ms        | 25.8, 25.3 ms   | 38.0, 18.5 ms    |
| Moving Bandwidth         | 12.20, 12.20 GB/s| 9.92, 10.12 GB/s| 5.27, 10.80 GB/s |
| Total Tokens             | 432              | 432             | 495              |
| Token Generation Time    | 9.19 s           | 11.56 s         | 48.63 s          |
| Token Generation Speed   | 47.03 tokens/s   | 37.36 tokens/s  | 10.18 tokens/s   |


```bash
nsys profile -o profile_report torchrun --nproc_per_node 1 copy_test.py \
--ckpt_dir /home/zinccat/Models/Llama-2-7b-chat \
    --tokenizer_path /home/zinccat/Models/Llama-2-7b-chat/tokenizer.model \
    --max_seq_len 512 --max_batch_size 1
```

using model from `/home/zinccat/Models/Llama-2-7b-chat`, please change it.
`path = '/home/zinccat/Models/Llama-2-7b-chat'`

## Code
### Run inference
Measure KV cache size of common generations.
```bash
torchrun --nproc_per_node 1 inference.py \
--ckpt_dir /home/zinccat/Models/Llama-2-7b-chat \
    --tokenizer_path /home/zinccat/Models/Llama-2-7b-chat/tokenizer.model \
    --max_seq_len 512 --max_batch_size 1
```

### Copy speed for KV cache
Check the speed of copying KV cache from CPU/GPU to GPU/CPU.
1. 
```bash
nsys profile -o profile_report torchrun --nproc_per_node 1 copy_test.py \
--ckpt_dir /home/zinccat/Models/Llama-2-7b-chat \
    --tokenizer_path /home/zinccat/Models/Llama-2-7b-chat/tokenizer.model \
    --max_seq_len 512 --max_batch_size 1
```
2. `nsys stats profile_report.nsys-rep` could reveil the copy speed.

### Generation speed
Measure generation speed.
```bash
torchrun --nproc_per_node 1 generation_speed.py \
--ckpt_dir /home/zinccat/Models/Llama-2-7b-chat \
    --tokenizer_path /home/zinccat/Models/Llama-2-7b-chat/tokenizer.model \
    --max_seq_len 512 --max_batch_size 1
```