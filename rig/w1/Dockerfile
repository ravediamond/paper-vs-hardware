# The W1 world-model environment of paper-vs-hardware (the 12 W1 / W2 posts): the published LeRobot image + the world-model
# stack + the transformers version that saved W1. Mirrors what the research rig ran (lerobot-jetson-wm:rc plus the
# transformers 5.19.0 overlay first on PYTHONPATH). Jetson Orin (aarch64, JetPack 6.2).
#   docker build -t ghcr.io/ravediamond/lerobot-jetson:w1 rig/w1
#   docker run --rm --runtime nvidia -v $PWD:/work -w /work ghcr.io/ravediamond/lerobot-jetson:w1 python3 posts/<folder>/scripts/<script>.py ...
FROM ghcr.io/ravediamond/lerobot-jetson:public
# every package already in the image is pinned, so pip can only add (the Jetson-built torch, torchvision, torchcodec and numpy stay)
RUN pip list --format=freeze > /tmp/image-constraints.txt \
 && pip install --no-cache-dir -c /tmp/image-constraints.txt \
      "stable-worldmodel[train,data] @ git+https://github.com/galilai-group/stable-worldmodel@21446f1ede6d5284e981bd7b47f432b994e6d812" \
      "stable-pretraining==0.1.8"
# W1 was saved under transformers 5.19.0, whose ViT key names differ from the image's 5.5.4 (see the voided-gates post):
# load it with 5.19.0, kept in its own directory first on the path, as on the rig. transformers 5.19.0 needs
# huggingface_hub >= the rig's 1.33.0 (the published base has 1.29.0), so the overlay carries it too.
RUN pip install --no-cache-dir --no-deps --target /opt/pydeps-w1 "transformers==5.19.0" "tokenizers==0.23.3" "huggingface_hub==1.33.0"
# the W1 posts' scripts import so101_harness.w1_live / .episodes: the needed subset, copied verbatim (so101_harness/__init__.py)
COPY so101_harness /opt/w1/so101_harness
COPY w1_load.py /opt/w1/w1_load.py
ENV PYTHONPATH=/opt/pydeps-w1:/opt/w1
RUN python -c "import transformers, stable_worldmodel, stable_pretraining, torch, so101_harness.w1_live, so101_harness.episodes; assert transformers.__version__ == '5.19.0', transformers.__version__; print('W1 env ok: transformers', transformers.__version__, 'torch', torch.__version__)"
