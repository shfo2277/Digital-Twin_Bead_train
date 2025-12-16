import tensorflow as tf
import platform
import subprocess

print("📌 Python 정보")
print("Python 버전:", platform.python_version())

print("\n📌 TensorFlow 정보")
print("TensorFlow 버전:", tf.__version__)
print("GPU 인식 여부:", tf.config.list_physical_devices('GPU'))

# 실제 연산 디바이스 로그 출력 활성화
print("\n📌 연산 디바이스 로그 활성화")
tf.debugging.set_log_device_placement(True)

# 실제로 연산이 GPU에서 수행되는지 확인
print("\n📌 실제 연산 테스트")
@tf.function
def test_op():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    return tf.matmul(a, b)

gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device):
    result = test_op()
    print("연산 결과:\n", result)

# NVIDIA 드라이버 및 GPU 상태 확인
print("\n📌 NVIDIA 드라이버 및 CUDA 상태 (nvidia-smi)")
try:
    output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    print(output)
except FileNotFoundError:
    print("❌ nvidia-smi 명령어를 찾을 수 없습니다.")
except Exception as e:
    print("❌ nvidia-smi 실행 오류:", e)
