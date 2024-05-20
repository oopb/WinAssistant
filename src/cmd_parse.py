import re


def extract_cmds_from_markdown(markdown_text):
    # 使用正则表达式匹配Markdown中的cmd指令
    # 假设cmd指令格式为：\[cmd\]\(指令内容\)
    cmd_pattern = r"```cmd(.*?)```"
    cmds = []
    for match in re.finditer(cmd_pattern, markdown_text, re.S):
        # 提取匹配到的cmd指令，并去除两端的空白字符
        cmd = match.group(1).strip()
        # 如果有多行cmd指令，就分割提取
        if '\n' in cmd:
            cmd_lines = cmd.split('\n')
            for line in cmd_lines:
                cmds.append(line.strip())
        else:
            cmds.append(cmd.strip())
    return cmds


# 示例Markdown文本
markdown_example = """
这是一个Markdown文本示例。

这里有一个cmd指令：[cmd](ls -l)

再试一个：[cmd](pwd)

```cmd
start explorer
```


```cmd

cd D:\test
echo #include <stdio.h> > test.c

```
请注意，这些指令会在方括号和括号中。
"""

# print(extract_cmds_from_markdown(markdown_example))
