# Setup and Training with CARLA

**NOTE:** Training this model with CARLA requires above average computational resources (around 20 GB of video ram) due to the complexity of the neural networks. If you have a sufficient computer (AMD or NVIDIA GPU with at least 20 GB of VRAM), follow the first instruction set, otherwise follow the second instruction set.

---
## Training the Model on Your Local Computer

1. Install CARLA 0.9.15 following [this guide.](https://carla.readthedocs.io/en/latest/start_quickstart/#b-package-installation)
2. Create and activate a virtual environment running Python 3.7 (latest version compatible with the CARLA python library).
4. Install the CARLA python library using the **.whl** file following the guide from step 1.
5. Install the PyTorch library compatible with Python 3.7 using the command below according to your GPU.

**NVIDIA GPU (CUDA 11.6)**
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
**AMD GPU (ROCM 5.2, Linux only)**
```
pip install torch==1.13.0+rocm5.2 torchvision==0.14.0+rocm5.2 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/rocm5.2
```

6. Install the rest of the project dependencies:
```
pip install -r requirements.txt
```

7. Run CARLA and wait until it is fully loaded (you can also run CARLA off-screen to save compute resources for training).
8. Run the training iterations:
```
python main.py
```

---
## Training the Model on the Cloud
1. Launch a VM instance on either [Lambda Labs](https://lambdalabs.com/service/gpu-cloud#pricing) or [Vast ai](https://vast.ai/?utm_source=googleads&utm_id=circleclick.com&gad_source=1&gclid=CjwKCAjw1emzBhB8EiwAHwZZxaH8av5HqdDSY_byXvA0UIg940bpkIkXW6ryxGg4NBf7d__-DuAzxxoChEAQAvD_BwE)
  - The instance should have a GPU with VRAM of at least 20 GB and an SSD/HD with at least 110 GB of storage.
  - The instance should have the Ubuntu operating system (version 20.04 or newer).
  - Note that you will need to create a private/public SSH key pair for connecting from your local terminal.
    - If you create the key pair from your local computer, you will need to change the permissions on your stored private key.
2. Clone this repository in your instance.
3. Make the shell script executable:
```
chmod +x path\to\this\project\MAPDDPG-CARLA\install_dependencies.sh
```
4. Run the shell script to install all necessary dependencies:
```
./path/to/shell/script/install_dependencies.sh
```
