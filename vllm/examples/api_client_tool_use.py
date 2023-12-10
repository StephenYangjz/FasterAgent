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


def post_http_request(api_url: str,
                      n: int = 1,
                      stream: bool = False,
                      query: str = None,
                      functions: List[dict] = None,
                      responses: List[dict] = None,
                      max_tokens: int = 1024,
                      temperature: float = 0.0) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    user_message = process_user_message(query)
    system_message = process_system_message(functions)
    new_functions = []
    for function, response in zip(functions, responses):
        function["call_info"] = response
        new_functions.append(function)
    functions = new_functions
    pload = {
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--n", type=int, default=1)
    # parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    query = '''I'm a researcher studying body fat percentage in individuals. Can you provide me with the necessary API to calculate body fat percentage based on imperial units? Also, fetch the hospitals' details for the nearest hospital to my university in Los Angeles.'''
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
        "response": {
            "content": ""
        }
    }]

    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream
    max_tokens = args.max_tokens

    # print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(api_url, n, stream, query=query, functions=functions, responses=responses, max_tokens=max_tokens)

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
