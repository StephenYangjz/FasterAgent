from vllm.agents.misc import get_conv_template
from typing import Dict, List, Tuple
import asyncio
import time
import torch
import logging

seed_value = 42
torch.manual_seed(seed_value)

class Function:
    def __init__(self, name: str, parameters: Dict, call_info: Dict) -> None:
        self.name = name
        self.parameters = parameters
        self.call_info = call_info

    def __str__(self) -> str:
        return f"Function: {self.name}\nParameters: {self.parameters}\nCall Info: {self.call_info}"

class APIInfo:
    def __init__(self, conversation_history: List[Dict], functions: List[Dict]) -> None:
        self.conversation_history = conversation_history
        self.function_info = {}
        self.task = None
        for function in functions:
            self.function_info[function["name"]] = Function(function["name"], function["parameters"], function["call_info"])
        self.response_tokens = None
        self.response_next = -1
    
    def __str__(self) -> str:
        return f"Conversation History: {self.conversation_history}\nFunction Info: {self.function_info}"

def flatten(conversation_history):
    template = "tool-llama-single-round"
    conv = get_conv_template(template)
    roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    prompt = ''
    for message in conversation_history:
        role = roles[message['role']]
        content = message['content']
        prompt += f"{role}: {content}\n"
    return prompt

def input_prompt(conversation_history):
    prompt = flatten(conversation_history)
    prompt += "Assistant:\n"
    return prompt

def output_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    action = [string[string.find("Action: ") + len("Action: "): string.find("\nAction Input: ")]]
    action_input = [string[string.find("Action Input: ") + len("Action Input: "):]]
    try:
        arguments = eval(action_input[0])
    except SyntaxError:
        arguments = {}
    except Exception as e:
        logging.error(f"{e}: {action_input[0]}")
        arguments = {}
    message = {
        "role": "assistant",
        "content": thought[0],
        "function_call": {
            "name": action[0],
            "arguments": arguments
        }
    }
    return message

def get_api_call(output_text: str, api_info: APIInfo, prompt_len: str) -> Tuple[bool, str, Dict]:
    last_message = output_text[len(input_prompt(api_info.conversation_history)) - prompt_len:]
    parsed_message = output_parser(last_message)
    api_info.conversation_history.append(parsed_message)
    function_name = parsed_message["function_call"]["name"]
    args_dict = parsed_message["function_call"]["arguments"]
    # TODO: check if we need function call
    call_api = (function_name != "Finish" and function_name in api_info.function_info)
    if call_api:
        # check if arguments are valid
        function = api_info.function_info[function_name]
        if function.parameters["required"] and not any (key in args_dict.keys() for key in function.parameters["required"]):
            # use logger to record error
            logging.error("Insufficient arguments")
            logging.error(f"Function: {function_name}")
            call_api = False
            
    return call_api, function_name, args_dict

class Task:
    def __init__(self):
        raise NotImplementedError
    def done(self):
        raise NotImplementedError
    def result(self):
        raise NotImplementedError


class Timer(Task):
    print("xxx")
    def __init__(self, delay: int, response: Dict, std = 0.1):
        self.done_time = delay / 1000 + time.monotonic() + torch.normal(mean=torch.Tensor([0.0]), std=torch.Tensor([std]))[0]
        self.response = response
    def done(self):
        return self.done_time <= time.monotonic()
    def result(self):
        return self.response

def call_api(function: Function, args_dict: Dict) -> asyncio.Task:
    if "response" in function.call_info:
        task = Timer(function.call_info["time"], function.call_info["response"])
    else:
        raise NotImplementedError
    return task

def main():
    x = {
        "name": "bodyfat_imperial_for_health_calculator_api",
        "description": '''This is the subfunction for tool \"health_calculator_api\", you can use this tool.The description of this function is: \"This endpoint calculates the body fat percentage based on the provided gender, age, height, and weight parameters in imperial units.\"''',
        "parameters": {
        "type": "object",
        "properties": {
            "age": {
            "type": "integer",
            "description": "The age of the person in **years**. Required.",
            "example_value": "26"
            },
            "weight": {
            "type": "integer",
            "description": "The weight of the person in **pounds**. Required.",
            "example_value": "180"
            },
            "gender": {
            "type": "string",
            "description": "The gender of the person. Must be either '**male**' or '**female**'. Required.",
            "example_value": "male"
            },
            "height": {
            "type": "integer",
            "description": "The height of the person in **inches**. Required.",
            "example_value": "77"
            }
        },
        "required": [
            "age",
            "weight",
            "gender",
            "height"
        ],
        "optional": []
        },  
        "call_info": {
            "time": 100,
            "response": {
                "role": "function",
                "name": "gethospitalsbyname_for_us_hospitals",
                "content": '''{"error": "", "response": "[{'Provider CCN': 50660, 'Hospital Name': 'USC Norris Comprehensive Cancer Center', 'Alias': 'Usc Norris Cancer Hospital', 'Url': 'http://www.uscnorriscancerhospital.org', 'Phone': '(323)865-3000', 'Service': 'Cancer', 'Type': 'Rehabilitation', 'Approval Codes': 'The Joint Commission accreditation,Cancer program approved by American College of Surgeons,Cancer program approved by American College of Surgeons,Member of Council of Teaching Hospitals of the Association of American Medical Colleges,,,,,Residency training approved by the Accreditation Council for Graduate Medical Education,,The Joint Commission accreditation,Medicare certification by the Centers for Medicare and Medicaid Services,Member of Council of Teaching Hospitals of the Association of American Medical Colleges,Medical school affiliation, reported to the American Medical Association', 'Street Address': '1441 Eastlake Avenue', 'City': 'Los Angeles', 'State Code': 'CA', 'Zip Code': '90089-0112', 'County': 'Los Angeles'...'''
            }
        },
    }

    function = Function(x["name"], x["parameters"], x["call_info"])
    task = call_api(function, None)
    now = time.monotonic()
    while not task.done():
        print(time.monotonic() - now)

    print(task.result())

if __name__ == '__main__':
    main()

