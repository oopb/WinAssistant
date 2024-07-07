import subprocess

import flet as ft
# from voice import call_with_stream
from tool_call import tool_call_run, messages, init_messages


def main(page: ft.Page):
    page.title = "大模型操作助手"
    page.window_height = 455
    page.window_width = 335
    page.window_resizable = False  # window is not resizable
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.MainAxisAlignment.CENTER
    order = ft.TextField(label="输入命令", autofocus=True, width=330, multiline=True, min_lines=2, max_lines=2)
    answer = ft.TextField(label="将要执行的命令", width=330, read_only=True, multiline=True, min_lines=2, max_lines=2,
                          disabled=True)

    def submit_click(e):
        if choose.value == "语音":
            answer.value = "开始聆听..."
            page.update()

            order.value = ""  # TODO 调用语音输入，返回一个用户命令字符串
            page.update()

        if not order.value:
            order.error_text = "输入为空"
            page.update()
        else:
            message = order.value
            answer.value = "正在思考中..."
            page.update()
            output = ""

            output = tool_call_run(message, messages)  # TODO 需要一个函数，输入为用户的命令(message)，返回值为LLM的回答。

            answer.value = output
            page.update()

    def refresh_click(e):
        init_messages(messages)

    # TODO 调用刷新函数，使大模型的记忆清除

    submit = ft.ElevatedButton("提交", on_click=submit_click)
    refresh = ft.ElevatedButton("刷新", on_click=refresh_click)
    model = ft.Text(value="当前模式：")
    choose = ft.Dropdown(
        options=[
            ft.dropdown.Option("文字"),
            ft.dropdown.Option("语音"),
        ],
        width=150
    )
    choose.value = "文字"
    row1 = ft.Row(
        [model, choose],
        spacing=80,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    row2 = ft.Row([submit, refresh], spacing=150)
    column = ft.Column(
        [
            row1,
            order,
            answer,
            row2
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=30
    )
    page.add(column)
    page.window_center()
    page.window_to_front()


if __name__ == "__main__":
    ft.app(target=main)
