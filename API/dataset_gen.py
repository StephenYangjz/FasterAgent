# generate new dataset for API task

import csv
import json
import re
from tqdm import tqdm
import ast
import random

FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""

TASK_DESCRIPTION = f'''You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:\n'''

def get_system_message():
    system_message = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
    system_message = system_message.replace("{task_description}", TASK_DESCRIPTION)
    assert "with a function call to actually excute your step." in system_message
    # we find that following ReACT format and merging the thought node and function call node is easier for model to learn to integrate the action input json string in its prediction than learn to predict a json string directly.
    system_message = system_message.replace("with a function call to actually excute your step.", "with a function call to actually excute your step. Your output should follow this format:\nThought:\nAction\nAction Input:\n")
    system_message = system_message + "\nSpecifically, you have access to the following APIs: "
    return system_message

def get_responses(filepath='api_response_data.csv'):
    # Initialize an empty list to store the dictionaries
    responses_dict = {}
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        # Iterate over each row in the CSV
        for row in csv_reader:
            name = row["API"]
            name = name.replace("-", "_")
            # Construct a dictionary for each row and append it to the list
            # response = {
            #     "time": int(row["Delay"]),
            #     "response": {
            #         "role": "function",
            #         "name": name,
            #         "content": row["Responses"]
            #     }
            # }
            responses = ast.literal_eval(row["Responses"])
            response = {
                "time": int(row["Delay"]),
                "content": responses
            }
            responses_dict[name] = response

    # Now responses_list contains all the dictionaries
    return responses_dict

response_dict = get_responses()

toolbench_datasetpath = '../train.json'
with open(toolbench_datasetpath, mode='r') as file:
    json_data = file.read()
data = json.loads(json_data)

system_message = get_system_message()
len_system_message = len(system_message)

pattern = r'for tool \"([^\"]+)\"'

flag = 0

query = []

for d in tqdm(data):
    legal = True
    for message in d.get('conversations', []):
        if message.get('from') == 'system':
            system_prompt = message.get('value')
            functions = system_prompt.split('Specifically, you have access to the following APIs: ')[1]
            functions = eval(functions)
            tools = []
            # check if functions in the response list
            for function in functions:
                if function.get('name') == 'Finish':
                    continue
                # Search for the pattern in the text
                match = re.search(pattern,function.get('description'))
                # Extract the matched substring if a match is found
                tool_name = match.group(1) if match else None
                function["parent_tool"] = tool_name
                tools.append(tool_name)
            tools = list(set(tools))
            times = {}
            responses = {}
            for tool in tools:
                # check if all tools in the response list
                if tool not in response_dict:
                    legal = False
                    break
                else:
                    times[tool] = response_dict[tool]['time']
                    responses[tool] = random.choice(response_dict[tool]['content'])

        if message.get('from') == 'user':
            user_prompt = message.get('value')
            user_prompt = user_prompt[1:-8]
            
            # save to list
            if legal:
                query.append({
                    'user_prompt': user_prompt,
                    'functions': functions,
                    'response': responses,
                    'times': times
                })
                flag += 1
            break
    if flag >= 50:
        break
random.shuffle(query)
# save to json
with open('api_query_data.json', 'w') as f:
    json.dump(query, f)