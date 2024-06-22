# Setup and Training with CARLA

**NOTE:** Training this model with CARLA requires above average computational resources (around 20 GB of video ram) due to the complexity of the neural networks. If you have a sufficient computer (AMD or NVIDIA GPU with at least 20 GB of VRAM), follow the first instruction set, otherwise follow the second instruction set.

---
## Training the Model on Your Local Computer

1. Install CARLA following [this guide.](https://carla.readthedocs.io/en/latest/start_quickstart/#b-package-installation)
2. Create and activate a virtual environment running Python 3.7 (latest version compatible with the CARLA python library).
4. Install the CARLA python library using the **.whkl** file following the guide from step 1.
5. Install the PyTorch library compatible with Python 3.7 using the command below that aligns with your GPU.

**NVIDIA GPU (CUDA 11.6)**
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
**AMD GPU ROCM 5.2 (Linux only)**
```
pip install torch==1.13.0+rocm5.2 torchvision==0.14.0+rocm5.2 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/rocm5.2
```
