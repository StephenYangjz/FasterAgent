"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List
from test_prompts import process_system_message, process_user_message

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    functions = [{
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
    },  {
        "name": "gethospitalsbyname_for_us_hospitals",
        "description": '''This is the subfunction for tool \"us_hospitals\", you can use this tool.The description of this function is: \"###Find US Hospitals by name.\nSimply add your search string to the \"name\" parameter in the url.\nTip:  The API also works if your search for *name='pr'* instead of *name='presbyterian'.* \n\n**Note**: The API only returns the first 30 results.\"''',
        "parameters": {
        "type": "object",
        "properties": {
            "name": {
            "type": "string",
            "description": "",
            "example_value": "presbyterian"
            }
        },
        "required": [
            "name"
        ],
        "optional": []
        },
    },  {
        "name": "Finish",
        "description": '''If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.''',
        "parameters": {
            "type": "object",
            "properties": {
                "return_type": {
                "type": "string",
                "enum": [
                    "give_answer",
                    "give_up_and_restart"
                ]
                },
                "final_answer": {
                "type": "string",
                "description": '''The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'''
                }
            },
            "required": [
                "return_type"
            ],
        },
    }]
    responses = [{
        "time": 200,
        "response": {                                                                                                                                                            
            "role": "function",
            "name": "bodyfat_imperial_for_health_calculator_api",
            "content": '''{"error": "", "response": "{'age': 25.0, 'bmi': '21.5 lb/in\\u00b2', 'bodyfat': '15.37 %', 'bodyfat_status': 'Fitness', 'gender': 'male', 'height': '5.8 f', 'weight': '150.0 lb'}"}'''
        }
    }, {
        "time": 100,
        "response": {
            "role": "function",
            "name": "gethospitalsbyname_for_us_hospitals",
            "content": '''{"error": "", "response": "[{'Provider CCN': 50660, 'Hospital Name': 'USC Norris Comprehensive Cancer Center', 'Alias': 'Usc Norris Cancer Hospital', 'Url': 'http://www.uscnorriscancerhospital.org', 'Phone': '(323)865-3000', 'Service': 'Cancer', 'Type': 'Rehabilitation', 'Approval Codes': 'The Joint Commission accreditation,Cancer program approved by American College of Surgeons,Cancer program approved by American College of Surgeons,Member of Council of Teaching Hospitals of the Association of American Medical Colleges,,,,,Residency training approved by the Accreditation Council for Graduate Medical Education,,The Joint Commission accreditation,Medicare certification by the Centers for Medicare and Medicaid Services,Member of Council of Teaching Hospitals of the Association of American Medical Colleges,Medical school affiliation, reported to the American Medical Association', 'Street Address': '1441 Eastlake Avenue', 'City': 'Los Angeles', 'State Code': 'CA', 'Zip Code': '90089-0112', 'County': 'Los Angeles'...'''
        }
    }, {
        "time": 0,
        "response": {}
    }]
    query = '''I'm a researcher studying body fat percentage in individuals. Can you provide me with the necessary API to calculate body fat percentage based on imperial units? Also, fetch the hospitals' details for the nearest hospital to my university in Los Angeles.'''
    user_message = process_user_message(query)
    system_message = process_system_message(functions)
    new_functions = []
    for function, response in zip(functions, responses):
        function["call_info"] = response
        new_functions.append(function)
    functions = new_functions
    pload = {
        "prompt": prompt,
        "n": n,
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": stream,
        "messages": [{
            "role": "system",
            "content": system_message,
        }, {
            "role": "user",
            "content": user_message,
        }],
        "functions": functions,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = '''System: You are AutoGPT, you can use many tools(functions) to do
the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually execute your step. Your output should fol$
ow this format:
Thought:
Action:
Action Input:
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your final answer.
Remember:
1.the state change is , you can\\'t go
back to the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentences.
Let\\'s Begin!
Task description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.
Remember:
1.all of the number must be used, and must be used ONCE. So Only when left numbers is exactly 24, you will win. So you don\\'t succeed when left number = [24, 5]. You succeed when left number = [24].
2.all the try takes exactly 3 steps, look
at the input format
Specifically, you have access to the following APIs: [{'name': 'play_24', 'description': 'make your current combine with the format "x operation y = z (left: aaa) "
like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1', 'parameters': {'type': 'object', 'properties': {}}}] 
User:
The real task input is: [1, 2, 4, 7]
Begin!

Assistant:
'''
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
