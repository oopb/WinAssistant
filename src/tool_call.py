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

agent = ZhipuAI(
    api_key=os.getenv("ZHIPUAI_API_KEY")
)

system_prompt = '''
你是一个本地Windows AI助手，能够使用工具完成用户要求，也能用户进行对话。
注意你可以调用工具"get_path"和"opr_cmd"来完成用户的要求。
"get_path"能搜索到文件的路径，
"opr_cmd"能执行cmd指令，和电脑应用文件进行交互。
注意：如果完成要求的cmd指令需要知道路径，可以先调用get_path来获取路径，再用opr_cmd来执行。
请根据历史上下文来判断现在是否应该调用工具，若需要工具调用请直接调用工具，生成tool类型的回答。
请逐步思考。
'''
# 注意如果，只是对话，不涉及到工具调用，请返回{arguments: { \"message\": \"你的回答\"}}

agent_prompt = '''
你是一个判断当前对话是否应该结束的agent。
你的职责是分析给定的聊天对话上下文，判断接下来是否应该由user输入指令。
对话的流程一般如下：
  1. user输入指令
  2. assistant分析指令
  3. tool被assistant调用
  4. assistant继续分析指令、调用工具直到执行完毕
  5. user输入指令，继续新一轮对话
你的目的是判断并预测出接下来是否应该进行到步骤5，如果是则输出"true"，否则输出"false"。
除此之外请不要输出其他任何东西。
'''

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
    # return os.system(cmd)


def agent_cmd(_messages):
    agent_message = [{
        "role": "system",
        "content": agent_prompt
    }, {
        "role": "user",
        "content": _messages
    }]
    res = agent.chat.completions.create(
        model="glm-4",
        messages=agent_message,
    )
    if res == "true":
        return True
    if res == "false":
        return False


def generate_response(_model, _messages, _tools, _max_tokens=500):
    res = _model.chat.completions.create(
        model="glm-4",
        messages=_messages,
        tools=_tools,
        tool_choice="auto",
        # temperature=0.2,
        # top_p=0.9,
        # stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=_max_tokens,
    )
    return res


tools_mapping = {
    "get_path": get_path,
    "opr_cmd": opr_cmd,
}


def parse_function_call(model_response, _messages):
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
        _messages.append({
            "role": "tool",
            "content": f"{json.dumps(function_result)}",
            "tool_call_id": tool_call.id
        })


messages = [{
    "role": "system",
    "content": system_prompt
}]


def tool_call_run(_message, _messages):
    question = _message
    _messages.append({
        "role": "user",
        "content": question
    })
    response = generate_response(client, _messages, tools)
    print(response.choices[0].message)
    _messages.append(response.choices[0].message.model_dump())
    while _messages[-1]["tool_calls"] is not None:
        # while agent_cmd(messages.__str__()):
        parse_function_call(response, _messages)
        response = generate_response(client, _messages, tools)
        print(response.choices[0].message)
        _messages.append(response.choices[0].message.model_dump())
    return _messages[-1]["content"]


def init_messages(_messages):
    _messages = [{
        "role": "system",
        "content": system_prompt
    }]

# while True:
#     res = tool_call_run(input("input:"), messages)
#     print(res)
