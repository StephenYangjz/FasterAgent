# Profiling

This folder is for profiling of common LLMs.

Usage:

1. First install PyTorch.
2. Install `llama` from local: `pip install -e .`
3. Follow instruction in Code part.

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
| nvbandwidth              | 25.00 GB/s       | 24.55 GB/s      | 26.33 GB/s       |
| Total Tokens             | 432              | 432             | 495              |
| Token Generation Time    | 9.19 s           | 11.56 s         | 48.63 s          |
| Token Generation Speed   | 47.03 tokens/s   | 37.36 tokens/s  | 10.18 tokens/s   |

Datapoints on 3090 (GtC, CtG (in ms)):
1. 256 tokens, 11.8, 11.6
2. 512 tokens, 21.3, 20.4
3. 1024 tokens, 39.4, 37.7
4. 2048 tokens, 75.8, 72.1
5. 2560 tokens, 273.8, 90.5
6. 3072 tokens, 325.1, 107.2
7. 4096 tokens, 590.4, 143.1


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

3. Use https://github.com/NVIDIA/nvbandwidth to check the bandwidth of GPU.

### Generation speed
Measure generation speed.
```bash
torchrun --nproc_per_node 1 generation_speed.py \
--ckpt_dir /home/zinccat/Models/Llama-2-7b-chat \
    --tokenizer_path /home/zinccat/Models/Llama-2-7b-chat/tokenizer.model \
    --max_seq_len 512 --max_batch_size 1
```