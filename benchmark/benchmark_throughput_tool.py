#!/usr/bin/env python3

import math
from enum import Enum
from transformers import AutoTokenizer
from typing import List
import aiohttp
import argparse
import asyncio
import itertools
import json
import os
# import random
import requests
import sys
import time
import numpy as np
import re
from test_prompts import process_system_message, process_user_message
from vllm.agents.utils import input_prompt


seed_value = 42
np.random.seed(seed_value)

def get_wait_time(mean_time_between_requests: float, distribution: str) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    else:
        return np.random.exponential(mean_time_between_requests)


def request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                time.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


async def async_request_gen(generator, qps: float, distribution="uniform"):
    while True:
        try:
            item = next(generator)
            yield item
            if distribution != "burst":
                await asyncio.sleep(get_wait_time(1.0 / qps, distribution))
        except StopIteration:
            return


class GenerationBackend(str, Enum):
    HfTextGenerationInference = "HfTextGenerationInference"
    vLLM = "vLLM"
    NaiveHfPipeline = "NaiveHfPipeline"
    RayGen = "RayGen"
    FasterTransformer = "FasterTransformer"


async def query_model_vllm(
    prompt,
    verbose,
    tokenizer,
    allow_variable_generation_length,
    total_requests,
    port,
    n: int = 1,
    stream: bool = False,
    query: str = None,
    functions: List[dict] = None,
    responses: List[dict] = None,
    max_tokens: int = 8192,
    temperature: float = 0,
):
    prompt, prompt_len, expected_response_len = prompt
    
    query = prompt['user_prompt']
    functions = prompt['functions']
    responses = prompt['response']
    times = prompt['times']

    headers = {"User-Agent": "Test Client"}
    user_message = process_user_message(query)
    system_message = process_system_message(functions)
    new_functions = []
    for function in functions:
        if function["name"] == "Finish":
            continue
        time = times[function["parent_tool"]]
        response = {"role": "function", "name": function["name"], "content": responses[function["parent_tool"]]}
        function["call_info"] = {"time": time, "response": response}
        new_functions.append(function)
    functions = new_functions
    messages = [{
                    "role": "system",
                    "content": system_message,
                }, 
                {
                    "role": "user",
                    "content": user_message,
                }]

    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # generate_input = dict(
        #     inputs=prompt,
        #     parameters=dict(
        #         prompt_len=prompt_len,
        #         reponse_len=expected_response_len,
        #     ),
        # )

        generate_input = {
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": messages,
            "functions": functions,
            "parameters": dict(
                prompt_len=prompt_len,
                reponse_len=expected_response_len,
            ),
        }

        if verbose:
            print("Querying model", len(system_message))
        async with session.post(
            f"http://localhost:{port}/generate", json=generate_input, headers=headers
        ) as resp:
            if verbose:
                print("Done")
            content = await resp.text()  # read the content as text (string)
            output = json.loads(content)
            # necessary for latency calc
            output["response_len"] = expected_response_len
            if verbose and "generated_text" in output:
                print(json.dumps(output["generated_text"]))

            return (input_prompt(messages), output)


def load_prompts(prompt_file):
    # with open(prompt_file) as f:
    #     prompts = [json.loads(l) for l in f.readlines()]
    # return prompts
    with open(prompt_file) as f:
        # read from json
        prompts = json.load(f)
    return prompts


def get_tok_id_lens(tokenizer, batch):
    tokenized = tokenizer.batch_encode_plus(batch)
    lens = [len(s) for s in tokenized["input_ids"]]
    return lens

def remove_api_result(response):
    # Define the pattern to be removed
    pattern = r"Function: .+?\nAssistant:\n"
    
    # Use regular expression to replace the pattern with an empty string
    cleaned_text = re.sub(pattern, '', response)

    return cleaned_text

def calculate_throughput(
    queries,
    dur_s,
    backend,
    tokenizer,
    median_token_latency,
    median_e2e_latency,
    all_e2e_latencies,
    all_per_token_latencies,
    results_filename,
    log_latencies,
    fail_on_response_failure,
    all_start_times = None
):
    prompts = []
    responses = []
    naive_hf_lens = []
    # ft_lens = []
    expected_response_lens = []
    ray_gen_lens = []
    # cf_gen_lens = []
    for prompt, response in queries:
        if "generated_text" in response:
            prompts.append(prompt)
            responses.append(remove_api_result(response["generated_text"]))
        if "naive_hf_lens" in response:
            naive_hf_lens.append(response["naive_hf_lens"])
        if "ray_gen_len" in response:
            ray_gen_lens.append(response["ray_gen_len"])
        # if "num_output_tokens_cf" in response:
        # removed due to incorrect calculation
        #     cf_gen_lens.append(response["num_output_tokens_cf"])

        if "response_len" in response:
            expected_response_lens.append(response["response_len"])
    prompt_ids = [p for p in tokenizer.batch_encode_plus(prompts)["input_ids"]]
    response_ids = [r for r in tokenizer.batch_encode_plus(responses)["input_ids"]]

    print(
        f"check_len actual {list(sorted(len(response) for response in response_ids))}"
    )
    print(f"check_len expect {list(sorted(expected_response_lens))}")
    # print(f"   self-reported {list(sorted(cf_gen_lens))}")

    for prompt, response, expected_response_len in zip(prompt_ids, response_ids, expected_response_lens):
       print(f'check lens {len(prompt)=} {len(response)=} {expected_response_len=}')

    try:
        prompt_lens = get_tok_id_lens(tokenizer, prompts)
        response_lens = get_tok_id_lens(tokenizer, responses)
    except Exception:
        print(prompts)
        print(responses)
        raise

    print(f"naive_hf_lens {list(sorted(naive_hf_lens))}")
    print(f"prompt_lens {list(sorted(prompt_lens))}")
    print(f"calc_throughput response_lens {list(sorted(response_lens))}")
    print(f"expected_response_lens {list(sorted(expected_response_lens))}")
    # print(f"ray_gen_lens {list(sorted(ray_gen_lens))}")

    prompt_token_count = sum(prompt_lens)
    response_token_count = sum(response_lens)

    if naive_hf_lens:
        # Manually count naive hf tok len
        total_resp_tokens = sum([response_len for _, response_len in naive_hf_lens])
        total_prompt_tokens = sum([prompt_len for prompt_len, _ in naive_hf_lens])

        response_token_count = total_prompt_tokens + total_resp_tokens

    if ray_gen_lens:
        response_token_count = sum(ray_gen_lens)

    if backend == GenerationBackend.NaiveHfPipeline:
        # It returns the prompt in the output.
        prompt_token_count = 0

    if backend == GenerationBackend.FasterTransformer:
        response_token_count = sum(expected_response_lens)

    # if cf_gen_lens:
    #     response_token_count = sum(cf_gen_lens)

    # print(f'prompt_token_count {prompt_token_count} response_token_count {response_token_count}')

    old_throughput_tok_s = (prompt_token_count + response_token_count) / dur_s
    # print(f'throughput_tok_s {throughput_tok_s:.02f}')

    mid_time = all_start_times[-20] - all_start_times[20] + all_e2e_latencies[-20]
    all_tokens = 0
    for i in range(100):
        if all_start_times[i] + all_e2e_latencies[i] < all_start_times[-20] + all_e2e_latencies[-20] and all_start_times[i] + all_e2e_latencies[i] > all_start_times[20]:
            all_tokens += prompt_lens[i] + response_lens[i]
    throughput_tok_s = all_tokens / mid_time
    qps = len(responses) / dur_s

    mean_e2e_latency = np.mean(all_e2e_latencies)

    # 90% tail latency
    tail_latency = np.percentile(np.array(all_e2e_latencies), 90)

    with open(results_filename, "a") as f:
        msg = f"backend {backend} dur_s {dur_s:.02f} real_throughput {throughput_tok_s:.02f} tokens_per_s {old_throughput_tok_s:.02f} qps {qps:.02f} successful_responses {len(responses)} prompt_token_count {prompt_token_count} response_token_count {response_token_count}, {median_token_latency=}, {median_e2e_latency=}, {mean_e2e_latency=}, 90_{tail_latency=}"
        if log_latencies:
            msg += f" {all_e2e_latencies=} {all_per_token_latencies=}"
        print(msg, file=f)
        print(msg)

    if fail_on_response_failure:
        assert (
            len(responses) == len(queries)
        ), f"{fail_on_response_failure=}, expected number of successful respones to equal number of queries, got {len(responses)} vs {len(queries)}"


def calculate_cdf(latencies):
    hist, bin_edges = np.histogram(latencies)
    cumsum = np.cumsum(hist)
    print(f"{bin_edges=}")
    print(f"{hist=}")
    print(f"{cumsum=}")


class MeasureLatency:
    def __init__(self):
        self._latencies = []
        self._per_token_latencies = []
        self._start_time = []

    def measure(self, f):
        async def measured(*args, **kwargs):
            start = time.time()
            self._start_time.append(start)
            prompt, output = await f(*args, **kwargs)

            # Do not record latency if request failed.
            if "generated_text" in output:
                latency = time.time() - start
                print(latency)
                self._latencies.append(latency)
                try:
                    self._per_token_latencies.append(latency / output["response_len"])
                except ZeroDivisionError:
                    # Not currently using this metric..
                    pass

            return prompt, output

        return measured


def get_token_ids(input_str, tokenizer):
    t = tokenizer(input_str)
    return t["input_ids"]


async def benchmark(
    backend: GenerationBackend,
    tokenizer,
    prompts: List[str],
    allow_variable_generation_length: bool,
    verbose: bool,
    results_filename: str,
    port: int,
    distribution: str,
    qps: float,
    log_latencies: bool,
    fail_on_response_failure: bool,
    max_tokens: int,
):
    if backend == GenerationBackend.vLLM:
        query_model = query_model_vllm
    else:
        raise ValueError(f"unknown backend {backend}")

    m = MeasureLatency()

    query_model = m.measure(query_model)

    if distribution == "burst":
        qps = float("inf")

    print(
        f"Starting with backend={backend}, num_prompts={len(prompts)}, allow_variable_generation_length={allow_variable_generation_length}"
    )
    print(f"traffic distribution={distribution}, qps={qps}")

    total_requests = len(prompts)

    async_prompts = async_request_gen(iter(prompts), qps=qps, distribution=distribution)

    start_time = time.time()
    tasks = []
    async for prompt in async_prompts:
        tasks.append(
            asyncio.create_task(
                query_model(
                    prompt,
                    verbose,
                    tokenizer,
                    allow_variable_generation_length,
                    total_requests,
                    port,
                    max_tokens=max_tokens,
                )
            )
        )
    queries = await asyncio.gather(*tasks)
    dur_s = time.time() - start_time

    median_token_latency = np.median(m._per_token_latencies)
    median_e2e_latency = np.median(m._latencies)
    all_start_times = np.array(m._start_time)
    # print(sorted(m._latencies))

    calculate_throughput(
        queries,
        dur_s,
        backend,
        tokenizer,
        median_token_latency,
        median_e2e_latency,
        m._latencies,
        m._per_token_latencies,
        results_filename,
        log_latencies,
        fail_on_response_failure,
        all_start_times
    )
    calculate_cdf(m._latencies)


def gen_random_response_lens(distribution: str, len_mean, len_range, num_prompts):
    if distribution == "uniform":
        if len_range == 0:
            return [len_mean for _ in range(num_prompts)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        num_to_generate = list(
            map(lambda _: np.random.randint(low, high), range(num_prompts))
        )
        return num_to_generate
    elif distribution == "exponential":
        # np.random.seed(random.randint(0, 1e6))
        return [
            min(round(s), len_range)
            for s in np.random.exponential(scale=len_mean, size=num_prompts)
        ]
    elif distribution == "capped_exponential":
        # np.random.seed(random.randint(0, 1e6))
        response_lens = []
        while len(response_lens) < num_prompts:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range:
                response_lens.append(sample)
        return response_lens
    else:
        raise ValueError(f"unknown distribution {distribution=}")


def gen_random_prompts(
    tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude=[]
):
    prompts, _ = gen_random_prompts_return_lens(
        tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude
    )
    return prompts


def gen_random_prompts_return_lens(
    tokenizer, len_mean, len_range, num_prompts, vocab_ids_to_exclude=[]
):
    low = len_mean - (len_range // 2)
    high = len_mean + (len_range // 2)
    vocab_ids = list(set(tokenizer.get_vocab().values()) - set(vocab_ids_to_exclude))

    def gen_prompt_ids(length):
        return [np.random.randint(10, 50000) for _ in range(length)]

    prompt_lens = list(map(lambda _: np.random.randint(low, high), range(num_prompts)))
    prompts_as_ids = list(
        map(lambda prompt_len: gen_prompt_ids(prompt_len), prompt_lens)
    )
    prompts = list(map(lambda prompt_ids: tokenizer.decode(prompt_ids), prompts_as_ids))

    # Because tokens do not map 1:1 to words, sometimes we get more tokens than desired.
    # This removes the additional tokens by tokenizing the prompt and cutting off additional tokens.
    # Confusingly, it works with a single iteration per prompt.
    for i, (p, l) in enumerate(zip(prompts, prompt_lens)):
        encoded = tokenizer(p)["input_ids"]
        if len(encoded) > l:
            # I am not sure why l-1 works, but it does..
            encoded = encoded[: l - 1]
        decoded = tokenizer.decode(encoded)
        encoded = tokenizer(decoded)["input_ids"]
        assert (
            len(encoded) == l
        ), f"Expected prompt to contain exactly {l} tokens, got {len(encoded)=}"
        prompts[i] = decoded

    return prompts, prompt_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--backend",
        type=GenerationBackend,
        choices=[e.name for e in GenerationBackend],
        required=True,
    )
    parser.add_argument("--results_filename", type=str, default="log")
    parser.add_argument("--port", type=int, required=True)

    parser.add_argument("--random_prompt_lens_mean", type=int)
    parser.add_argument("--random_prompt_lens_range", type=int)
    parser.add_argument("--random_prompt_count", type=int)

    parser.add_argument(
        "--distribution", choices=["burst", "uniform", "poisson"], default="burst"
    )
    parser.add_argument("--qps", type=float, default=5.0)

    parser.add_argument(
        "--log_latencies",
        action="store_true",
        help="Whether or not to write all latencies to the log file.",
    )
    parser.add_argument(
        "--fail_on_response_failure",
        action="store_true",
        help="Whether or not to fail the benchmarking script if any request fails",
    )

    parser.add_argument("--variable_response_lens_mean", type=int)
    parser.add_argument("--variable_response_lens_range", type=int)
    parser.add_argument(
        "--variable_response_lens_distribution",
        choices=["uniform", "exponential", "capped_exponential"],
        default="uniform",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompts_filename", type=str)
    group.add_argument("--gen_random_prompts", action="store_true")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--allow_variable_generation_length", action="store_true")
    group.add_argument("--fixed_max_tokens", type=int, default=256)
    group.add_argument("--max_tokens", type=int, default=8192)

    parser.add_argument("--print-generation-lens-and-exit", action="store_true")

    args = parser.parse_args()

    if args.gen_random_prompts:
        assert args.random_prompt_count is not None

    backend = GenerationBackend[args.backend]
    tokenizer = AutoTokenizer.from_pretrained("ToolBench/ToolLLaMA-2-7b-v2") #hf-internal-testing/llama-tokenizer") #ToolBench/ToolLLaMA-2-7b-v2")

    if args.prompts_filename:
        prompts = load_prompts(args.prompts_filename)
        prompt_lens = itertools.repeat(-1)
        # maybe calculate it
        num_prompts = len(prompts)
    elif args.gen_random_prompts:
        num_prompts = args.random_prompt_count
        random.seed(0xCADE)
        prompts, prompt_lens = gen_random_prompts_return_lens(
            tokenizer,
            len_mean=args.random_prompt_lens_mean,
            len_range=args.random_prompt_lens_range,
            num_prompts=num_prompts,
            vocab_ids_to_exclude=tokenizer.all_special_ids,
        )
    else:
        raise ValueError("unknown prompts")

    # print(len(prompts[0]))
    # print(prompts[0][0])
    # exit(0)

    if args.allow_variable_generation_length:
        response_lens = gen_random_response_lens(
            args.variable_response_lens_distribution,
            args.variable_response_lens_mean,
            args.variable_response_lens_range,
            num_prompts=num_prompts,
        )
        args.fixed_max_tokens = -1
    else:
        response_lens = [args.fixed_max_tokens for _ in range(num_prompts)]

    for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
        total = prompt_len + gen_len
        if total > 8192:
            print(f"truncating long prompt+gen_len {prompt_len=} {gen_len=}")
            gen_len = 8192 - prompt_len
        response_lens[i] = gen_len

    if args.print_generation_lens_and_exit:
        print(f"{prompt_lens=}")
        print(f"{response_lens=}")
        print("Exiting...")
        return

    if args.verbose or True:
        print("prompt lens", prompt_lens)
        print("response lens", response_lens)

        totals = []
        for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
            totals.append(prompt_len + gen_len)

        print("total tokens", list(sorted(totals)))

    prompts = list(zip(prompts, prompt_lens, response_lens))

    asyncio.run(
        benchmark(
            backend,
            tokenizer,
            prompts,
            args.allow_variable_generation_length,
            args.verbose,
            args.results_filename,
            args.port,
            args.distribution,
            args.qps,
            args.log_latencies,
            args.fail_on_response_failure,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
