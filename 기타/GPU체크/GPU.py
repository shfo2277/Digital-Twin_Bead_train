#import tensorflow as tf
import torch
import platform
import subprocess
import sys

# print("📌 Python 정보")
# print("Python 버전:", platform.python_version())

# print("\n📌 TensorFlow 정보")
# print("TensorFlow 버전:", tf.__version__)
# print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))

# print("\n📌 PyTorch 정보")
# print("PyTorch 버전:", torch.__version__)
# print("CUDA 사용 가능:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("사용 중인 GPU:", torch.cuda.get_device_name(0))
#     print("CUDA 버전 (torch):", torch.version.cuda)
#     print("cuDNN 버전 (torch):", torch.backends.cudnn.version())

# print("\n📌 NVIDIA 드라이버 및 CUDA 확인 (nvidia-smi)")
# try:
#     output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
#     print(output)
# except FileNotFoundError:
#     print("❌ nvidia-smi 명령어를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있지 않을 수 있습니다.")
# except Exception as e:
#     print("nvidia-smi 실행 중 오류:", e)


import platform
import subprocess
import sys

print("Python 정보")
print("Python 버전:", platform.python_version())

# # TensorFlow 확인
# try:
#     import tensorflow as tf
#     print("\n📌 TensorFlow 정보")
#     print("TensorFlow 버전:", tf.__version__)
#     print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))
# except ImportError:
#     print("\n📌 TensorFlow 정보")
#     print("❌ TensorFlow가 설치되어 있지 않습니다.")

# PyTorch 확인
try:
    import torch
    print("\n PyTorch 정보")
    print("PyTorch 버전:", torch.__version__)
    print("torchvision 버전:", __import__('torchvision').__version__)
    print("CUDA 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("사용 중인 GPU:", torch.cuda.get_device_name(0))
        print("CUDA 버전 (torch):", torch.version.cuda)
        print("cuDNN 버전 (torch):", torch.backends.cudnn.version())
except ImportError:
    print("\n PyTorch 정보")
    print("!!!!!!! PyTorch가 설치되어 있지 않습니다.")

# NVIDIA 드라이버 및 CUDA
print("\n NVIDIA 드라이버 및 CUDA 확인 (nvidia-smi)")
try:
    output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    print(output)
except FileNotFoundError:
    print(" nvidia-smi 명령어를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있지 않을 수 있습니다.")
except Exception as e:
    print("nvidia-smi 실행 중 오류:", e)

# 최종 종합 판단
print("\n@@@@@ 최종 GPU 사용 가능 여부 요약")
gpu_tf = False
gpu_torch = False

try:
    if tf.config.list_physical_devices('GPU'):
        gpu_tf = True
except:
    pass

try:
    if torch.cuda.is_available():
        gpu_torch = True
except:
    pass

if gpu_tf or gpu_torch:
    print("🎉 GPU를 사용할 수 있습니다.")
    if gpu_tf:
        print("✔ TensorFlow에서 GPU 사용 가능")
    if gpu_torch:
        print("✔ PyTorch에서 GPU 사용 가능")
else:
    print("⚠ GPU를 사용할 수 없습니다. 설치 또는 환경 구성을 확인하세요.")
