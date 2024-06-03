import json
import os

from dotenv import load_dotenv, find_dotenv

from es.everything_test import my_query
from run import run_cmd
from zhipuai import ZhipuAI

_ = load_dotenv(find_dotenv())

client = ZhipuAI(
    api_key=os.getenv("ZHIPUAI_API_KEY")
)

system_prompt = '''
你是一个本地Windows AI助手，能够使用工具完成用户要求，也能用户进行对话。
注意你可以调用工具"get_path"和"opr_cmd"来完成用户的要求。
"get_path"能搜索到文件的路径，
"opr_cmd"能执行cmd指令，和电脑应用文件进行交互。
注意：如果完成要求的cmd指令需要知道路径，可以先调用get_path来获取路径，再用opr_cmd来执行。
请根据历史上下文来判断现在是否应该调用工具，若需要两次工具调用请直接调用工具。
请逐步思考。
'''
# 注意如果，只是对话，不涉及到工具调用，请返回{arguments: { \"message\": \"你的回答\"}}

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


def get_path(file):
    res = my_query(file)
    return res[:5]


def opr_cmd(cmd):
    return run_cmd(cmd)


def generate_response(_model, _messages, _tools, _max_tokens=500):
    response = _model.chat.completions.create(
        model="glm-4",
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
    "opr_cmd": opr_cmd,
}


def parse_function_call(model_response, messages):
    # 处理函数调用结果，根据模型返回参数，调用对应的函数。
    # 调用函数返回结果后构造tool message，再次调用模型，将函数结果输入模型
    # 模型会将函数调用结果以自然语言格式返回给用户。
    if model_response.choices[0].message.tool_calls:
        tool_call = model_response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        function_result = {}
        # if tool_call.function.name == "get_path":
        #     function_result = get_path(**json.loads(args))
        # if tool_call.function.name == "opr_cmd":
        #     function_result = opr_cmd(**json.loads(args))
        if tool_call.function.name in tools_mapping:
            function_result = tools_mapping[tool_call.function.name](**json.loads(args))
        messages.append({
            "role": "tool",
            "content": f"{json.dumps(function_result)}",
            "tool_call_id": tool_call.id
        })


messages = [{
    "role": "system",
    "content": system_prompt
}]
while True:
    question = input("请输入：")
    messages.append({
        "role": "user",
        "content": question
    })
    response = generate_response(client, messages, tools)
    print(response.choices[0].message)
    messages.append(response.choices[0].message.model_dump())
    while messages[-1]["tool_calls"] is not None:
        parse_function_call(response, messages)
        response = generate_response(client, messages, tools)
        print(response.choices[0].message)
        messages.append(response.choices[0].message.model_dump())
