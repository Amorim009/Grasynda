param(
    [string]$PythonExe = "python"
)

Write-Host "Universal GPU preflight..."
$env:TF_CPP_MIN_LOG_LEVEL = "3"

$preflight = @'
import sys

try:
    import torch
except Exception as exc:
    print(f'PyTorch import failed: {exc}')
    sys.exit(1)

torch_cuda = torch.cuda.is_available()
print(f'torch_version={torch.__version__}')
print(f'torch_cuda_available={torch_cuda}')
print(f'torch_cuda_version={torch.version.cuda}')
print(f'torch_cuda_device_count={torch.cuda.device_count() if torch_cuda else 0}')
if torch_cuda:
    print(f'torch_cuda_device_name={torch.cuda.get_device_name(0)}')

try:
    import tensorflow as tf
    tf_gpus = tf.config.list_physical_devices('GPU')
    print(f'tensorflow_built_with_cuda={tf.test.is_built_with_cuda()}')
    print(f'tensorflow_gpu_count={len(tf_gpus)}')
except Exception as exc:
    print(f'TensorFlow probe failed: {exc}')

if not torch_cuda:
    print('ERROR: GPU runner requires a CUDA-enabled PyTorch installation.')
    sys.exit(2)
'@

& $PythonExe -u -c $preflight
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonExe -u "run_universal_experiments_gpu.py"
exit $LASTEXITCODE
