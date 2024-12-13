import json

from llama_cpp import Llama
import re
import subprocess
from everytools import EveryTools
es = EveryTools()

model = Llama(
    "D:/Llama3/Llama3-8B-Chinese-Chat-q8-v2.gguf",
    verbose=False,
    chat_format="chatml-function-calling", #
    n_ctx=8192,
    n_gpu_layers=-1,
)

system_prompt = ("你是一个本地Windows AI助手，能够使用工具完成用户要求，也能用户进行对话。\
注意你可以调用工具\"get_path\"和\"opr_cmd\"来完成用户的要求。\
\"get_path\"能搜索到文件的路径，\
\"opr_cmd\"能执行cmd指令，和电脑应用文件进行交互。 \
注意：如果完成要求的cmd指令需要知道路径，可以先调用get_path来获取路径，再用opr_cmd来执行。 \
请逐步思考。\
")
# 注意如果，只是对话，不涉及到工具调用，请返回{arguments: { \"message\": \"你的回答\"}}

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

def opr_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = result.stdout
    stderr = result.stderr
    out = []
    if result.returncode == 0:
        cmdout = {}
        cmdout["out"] = stdout
        out.append(cmdout)
        print("命令执行成功，输出为：")
        print(stdout)
    else:
        cmderr = {}
        cmderr["err"] = stderr
        out.append(cmderr)
        print("命令执行失败，错误信息为：")
        print(stderr)
    return json.dumps(out)

def generate_response(_model, _messages, _tools, _max_tokens=100):
    response = _model.create_chat_completion(
        messages=_messages,
        tools=_tools,
        tool_choice="auto",
        # temperature=0.2,
        # top_p=0.9,
        # stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=_max_tokens,
    )
    return response

tools_mapping = {
    "get_path": get_path,
    "opr_cmd": opr_cmd
}

def interact_chat():
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]
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
    user_input = input()
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    while True:
        response = generate_response(model, messages, tools)
        res_content = response["choices"][0]["message"]["content"]
        tool_calls = response["choices"][0]["message"].get("tool_calls", None)
        messages.append(response["choices"][0]["message"])
        for tool in tool_calls:
            tool_name = tool["function"]["name"]
            tool_args = json.loads(tool["function"]["arguments"])
            if tool_name in tools_mapping:
                tool_response = tools_mapping[tool_name](**tool_args)
                messages.append({
                    "role": "assistant",
                    "content": tool_response,
                })
            else:
                print("Windows Assistant: ", tool_args.get("message", None))
                user_input = input()
                messages.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )
        if user_input == "exit":
            break
interact_chat()

