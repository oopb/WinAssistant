import subprocess


def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(cmd, timeout=1)
    except subprocess.TimeoutExpired:
        stdout = process.stdout
        stderr = process.stderr
        process.kill()
    out = []
    if process.returncode == 0:
        cmd_out = {"out": "success"}
        out.append(cmd_out)
        print("命令执行成功，输出为：")
        print(stdout)
    else:
        cmd_err = {"err": "error"}
        out.append(cmd_err)
        print("命令执行失败，错误信息为：")
        print(stderr)
    return out

# print(run_cmd("start D:\\\\QQ\\\\Bin\\\\QQ.exe"))
# print("over")
