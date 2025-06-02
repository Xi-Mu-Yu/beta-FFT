import os
import platform

def auto_shutdown(delay=0):
    # 检测操作系统
    system_name = platform.system()

    if system_name == "Windows":
        # Windows 系统
        os.system(f"shutdown /s /t {delay}")
    elif system_name == "Linux" or system_name == "Darwin":  # Darwin 是 macOS
        # Linux 或 macOS 系统
        if delay == 0:
            os.system("shutdown -h now")
        else:
            os.system(f"shutdown -h +{delay // 60}")  # 将秒转换为分钟
    else:
        print("不支持的操作系统")

# 调用函数并设置延迟时间（秒）
auto_shutdown(delay=0)  # 立即关机
# python train_promise12_fft_20.py
# python shutd.py

# 1