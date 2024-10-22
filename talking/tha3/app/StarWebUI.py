import subprocess

diffusion_process = subprocess.Popen(
    ['D:/AIGC/绘世启动器/sd-webui-aki-v4.9/python/python.exe', 
     'D:/AIGC/绘世启动器/sd-webui-aki-v4.9/launch.py', 
     '--api',
     '--listen', '127.0.0.1',
     '--port', '8848',
     ],
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE
)
try:
    diffusion_process.wait()
except KeyboardInterrupt:
    print("Terminating the subprocess...")
    diffusion_process.terminate()