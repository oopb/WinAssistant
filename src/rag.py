import os

from dotenv import load_dotenv, find_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from run import run_cmd
from cmd_parse import extract_cmds_from_markdown

_ = load_dotenv(find_dotenv())

model_name = "BAAI/bge-large-zh"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# load data to Chroma db
db = Chroma(persist_directory="./chroma/document_pdf", embedding_function=embeddings)

llm_zhipu = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,
    api_key=os.getenv("ZHIPUAI_API_KEY"),
)

retriever = db.as_retriever(search_type="similarity")

contextualize_q_system_prompt = """
给定聊天历史记录和可能参考聊天历史记录中上下文的最新用户问题，制定一个独立的问题，\
该问题可以在没有聊天历史记录的情况下理解。\
不要回答这个问题，如果需要，只需重新制定，否则按原样返回。
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm_zhipu, retriever, contextualize_q_prompt
)

qa_system_prompt = \
    '''
        **任务描述**
        你是一个命令行指令翻译器，用于将用户的需求翻译为Windows11系统中的命令行语句，且答案可以直接在终端中运行。
        请根据用户输入的上下文回答问题，并遵守回答要求。

        **背景知识**
        {context}

        **回答要求**
        - 你输出的cmd指令的格式为：使用markdown中的代码块的格式。
        - 你回答的内容有且仅有指定格式的cmd命令行指令，禁止出现其他任何形式的自然语言文字，以确保你的答案符合命令行的语法要求，且能够直接在终端中正确运行。
        - 若存在多个可能的结果，只输出一个最符合要求的结果。
        - 对于与背景知识不相关的信息，直接回答“未找到相关答案”。
    '''

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "打开设置"),
        ("assistant", "```cmd\
         start ms-settings:\
         ```"),
        ("human", "打开环境变量"),
        ("assistant", "```cmd\
         rundll32 sysdm.cpl,EditEnvironmentVariables\
         ```"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm_zhipu, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

question = input("输入问题：")
while True:
    result = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "test_chat"}},
    )
    print(result['answer'])
    cmds = extract_cmds_from_markdown(result['answer'])
    for cmd in cmds:
        try:
            run_cmd(cmd)
            question = input("输入问题：")
        except Exception as e:
            # question = "你生成的cmd指令在终端运行之后有如下报错信息：\n" + str(e) + "\n请重新生成"
            question = input("输入问题：")

# chat_history = []
# while True:
#     question = input("输入问题：")
#     result = rag_chain.invoke(
#         {"input": question, "chat_history": chat_history},
#     )
#     print(result['answer'])
#     chat_history.extend([HumanMessage(content=question), result['answer']])
#     # print(result)
#     # print(chat_history)
#     # if len(chat_history) > 5:
#     #     chat_history = chat_history[-5:]
