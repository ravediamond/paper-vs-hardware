# Jetson (aarch64) image for LeWM smoke tests: lerobot-jetson:rc + the world-model stack. NOT the training environment
# (that is wm_train/Dockerfile, x86_64 + CUDA 12.8): this only proves the training code runs on a real GPU.
# Every package already in the base image is pinned as a constraint, so pip can only add packages: the Jetson-built
# torch / torchvision / torchcodec and numpy cannot be replaced (the build fails instead).
#   docker build -t lerobot-jetson-wm:rc -f wm_train/jetson/Dockerfile wm_train/jetson
FROM lerobot-jetson:rc
RUN pip list --format=freeze > /tmp/image-constraints.txt \
 && pip install --no-cache-dir -c /tmp/image-constraints.txt \
      "stable-worldmodel[train,data] @ git+https://github.com/galilai-group/stable-worldmodel@21446f1ede6d5284e981bd7b47f432b994e6d812" \
      "stable-pretraining==0.1.8" \
 && python -c "import torch, numpy, stable_worldmodel, stable_pretraining, lightning; \
from stable_worldmodel.data import LeRobotAdapter; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available() or 'n/a at build', 'numpy', numpy.__version__, 'lightning', lightning.__version__)"
