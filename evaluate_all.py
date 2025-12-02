import subprocess
import sys
import os

# 定义要保存结果的日志文件
LOG_FILE = "evaluation_results.txt"

# 定义需要按顺序运行的三个命令
COMMANDS = [
    "python tqa_tf_ensembler.py",
    "python tqa_ndq_ensembler.py",
    "python tqa_dq_ensembler.py"
]

def run_and_log(cmd, log_handle):
    """
    运行命令，并将其输出同时打印到屏幕和写入日志文件。
    强制使用 utf-8 编码读取，防止 Windows 下出现 GBK 解码错误。
    """
    header = f"\n{'='*50}\nStarting command: {cmd}\n{'='*50}\n"
    print(header)
    log_handle.write(header)
    log_handle.flush()

    # 使用 subprocess 启动进程
    # 新增: encoding='utf-8' 和 errors='replace' 以解决编码问题
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1, # 行缓冲
        encoding='utf-8',
        errors='replace'
    )

    # 实时读取输出
    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line) # 打印到屏幕
            log_handle.write(line) # 写入文件
            sys.stdout.flush()
            log_handle.flush()

    return process.wait()

def main():
    # 如果日志文件已存在，先清空它
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation started...\n")

    # 以追加模式打开文件，开始执行命令
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        for cmd in COMMANDS:
            return_code = run_and_log(cmd, f)
            if return_code != 0:
                error_msg = f"\nERROR: Command '{cmd}' failed with return code {return_code}.\nStopping execution.\n"
                print(error_msg)
                f.write(error_msg)
                sys.exit(return_code)

    print(f"\nAll commands finished successfully. Results saved to {LOG_FILE}")

if __name__ == "__main__":
    main()