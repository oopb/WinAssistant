from llama_cpp import Llama
from everytools import EveryTools

import re
import json
import subprocess
model = Llama(
    "D:/Llama3/Llama3-8B-Chinese-Chat-q8-v2.gguf",
    verbose=False,
    n_ctx=8192,
    n_threads=4,
    n_gpu_layers=-1,
)

system_prompt = "你是一个ai助手,你可以调用"

es = EveryTools()


def generate_reponse(_model, _messages, _max_tokens=100):
    _output = _model.create_chat_completion(
        _messages,
        temperature=0.2,
        top_p=0.9,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=_max_tokens,
    )["choices"][0]["message"]["content"]
    return _output

def interact_chat():
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]
    while True:
        user_input = input()
        if user_input == "exit":
            break
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        response = generate_reponse(model, messages)
        print("Windows Assistant: ", response)

        messages.append(response)


def get_path(file):
    es.search(file)
    res = es.results()
    rows = []
    for index, r in res.iterrows():
        row = {}
        for label, value in r.items():
            if label == "path" or label == "name":
                row[label] = value
        rows.append(row)
    return json.dumps(rows[:3])


if __name__ == '__main__':
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_path",
                "description": "Get the path of the file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "The name of the file, e.g. QQ.exe",
                        }
                    },
                    "required": ["file"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "opr_cmd",
                "description": "Operate the cmd command and get the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The windows cmd command, e.g. start QQ.exe",
                        }
                    },
                    "required": ["cmd"],
                },
            }
        }
    ]

    # Debugging output to check the structure of tools
    print("Tools:", tools)

    # Generate the list of function names and join them
    try:
        # function_names = " | ".join(
        #     [f'''functions.{tool['function']['name']}''' for tool in tools]
        # )
        function_names = " | ".join(
            [f'''"functions.{tool['function']['name']}:"''' for tool in tools]
        )
        print("Function Names:", function_names)
    except TypeError as e:
        print("Encountered a TypeError:", e)
    except KeyError as e:
        print("Encountered a KeyError:", e)

    # Output the generated function names
    print("Generated function names:", function_names)