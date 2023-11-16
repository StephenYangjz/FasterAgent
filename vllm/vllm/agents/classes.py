from typing import List

class APIPrompt:
    """Class for API prompt.
    Define:
        - prompt (str): the prompt to use for the generation.
        - API name (List[str]): the API name. 
        - API calling time (List[float]): the API calling time.
        - API return length (List[int]): the API return length.
    """

    def __init__(self, prompt: str, api_name: List[str], api_calling_time: List[float], api_return_length: List[int]):
        self.prompt = prompt
        self.api_name = api_name
        self.api_calling_time = api_calling_time
        self.api_return_length = api_return_length