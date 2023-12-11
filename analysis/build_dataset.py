import json
from tqdm import tqdm
from vllm.agents.misc import get_conv_template
from transformers import AutoTokenizer

def flatten(conversation_history):
    template = "tool-llama-single-round"
    conv = get_conv_template(template)
    roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    prompt = ''
    for message in conversation_history:
        role = roles[message['from']]
        content = message['value']
        prompt += f"{role}: {content}\n"
    return prompt

with open('/home/zinccat/datasets/data/data/toolllama_G123_dfs_train.json', 'r') as file:
    data = json.load(file)

# data = pd.DataFrame(data)

dataset_new = []

tokenizer = AutoTokenizer.from_pretrained("ToolBench/ToolLLaMA-2-7b-v2")

for idx, conv in tqdm(enumerate(data)):
    prompt = flatten(conv['conversations'][:2])
    prompt = prompt.split('You have access of the following tools:\n')[1].split('\n\nSpecifically,')[0]
    num_api_calls = 0
    num_tokens = 0
    first_response = True
    count = True
    userprompt = conv['conversations'][1]['value'][1:-8]
    userprompt_len = len(tokenizer.batch_encode_plus([userprompt])["input_ids"][0])
    legal = False
    for message in conv['conversations'][2:]:
        if message['from'] == 'function':
            num_api_calls += 1
        if message['from'] == 'user':
            # do not allow more user interference
            userprompt_len = len(tokenizer.batch_encode_plus([userprompt])["input_ids"][0])
            count = False
            break
        if message['from'] == 'assistant':
            ll = len(tokenizer.batch_encode_plus([message['value']])["input_ids"][0])
            num_tokens += ll
            if first_response:
                first_response = False
                first_response_length = ll
            if 'give_answer' in message['value']:
                legal = True
                num_tokens_give_answer = ll
    if legal and userprompt_len > 0 and count: #count and num_api_calls > 0:
        dataset_new.append({'prompt': prompt + '\n' + userprompt, 'userprompt': userprompt, 'num_api_calls': num_api_calls, 'num_tokens': num_tokens, 'prompt_length': len(tokenizer.batch_encode_plus([prompt])["input_ids"][0]), 'first_response_length': first_response_length, 'num_tokens_give_answer': num_tokens_give_answer, 'userprompt_length': userprompt_len})
    # if idx == 10000:
    #     break

# save to json
with open('/home/zinccat/datasets/data/data/toolllama_G123_dfs_train_tokenlength_legal.json', 'w') as file:
    json.dump(dataset_new, file, indent=4)

print(len(dataset_new))