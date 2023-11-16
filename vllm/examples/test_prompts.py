


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

FORMAT_INSTRUCTIONS_USER_FUNCTION = """
{input_description}
Begin!
"""

FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT = """Answer the following questions as best you can. Specifically, you have access to the following APIs:

{func_str}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of {func_list}
Action Input: the input to the action
End Action

Begin! Remember: (1) Follow the format, i.e,
Thought:
Action:
Action Input:
End Action
(2)The Action: MUST be one of the following:{func_list}
(3)If you believe that you have obtained enough information (which can be judge from the history observations) that can answer the task, please call:
Action: Finish
Action Input: {{"return_type": "give_answer", "final_answer": your answer string}}.
Question: {question}

Here are the history actions and observations:
"""

TASK_DESCRIPTION = f'''You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:\n'''

def process_system_message(functions):
    system_message = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
    system_message = system_message.replace("{task_description}", TASK_DESCRIPTION)
    assert "with a function call to actually excute your step." in system_message
    # we find that following ReACT format and merging the thought node and function call node is easier for model to learn to integrate the action input json string in its prediction than learn to predict a json string directly.
    system_message = system_message.replace("with a function call to actually excute your step.", "with a function call to actually excute your step. Your output should follow this format:\nThought:\nAction\nAction Input:\n")
    # add all the function dicts in the prompt.
    system_message += str(functions)
    # system_message = system_message + "\nSpecifically, you have access to the following APIs: " + str(functions)
    return system_message

def process_user_message(query):
    user_message = FORMAT_INSTRUCTIONS_USER_FUNCTION
    user_message = user_message.replace("{input_description}", query)
    return user_message

        