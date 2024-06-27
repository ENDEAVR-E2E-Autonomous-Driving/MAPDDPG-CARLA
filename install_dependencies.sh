#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Update package list and install prerequisites
sudo apt-get update
sudo apt-get install -y wget bzip2

# Create a new user for running CARLA (if not already created)
USERNAME="carla_user"
if id -u $USERNAME >/dev/null 2>&1; then
    echo "User $USERNAME already exists"
else
    sudo adduser --disabled-password --gecos "" $USERNAME
    sudo usermod -aG sudo $USERNAME
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/$USERNAME
fi

# Download and install Anaconda
ANACONDA_INSTALLER=Anaconda3-2023.03-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/$ANACONDA_INSTALLER
bash $ANACONDA_INSTALLER -b -p /home/$USERNAME/anaconda3

# Initialize Anaconda for the new user
sudo -u $USERNAME bash << EOF
eval "\$($HOME/anaconda3/bin/conda shell.bash hook)"
echo "source $HOME/anaconda3/bin/activate" >> ~/.bashrc

# Create and activate a conda environment
conda create -y -n carla_env python=3.7
conda activate carla_env

# install project dependencies
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r MAPDDPG-CARLA/requirements.txt

# Install CARLA and other dependencies
sudo apt-get -y install libomp5
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
mkdir carla-simulator
tar -xzvf CARLA_0.9.15.tar.gz -C carla-simulator/
pip3 install -r carla-simulator/PythonAPI/examples/requirements.txt

# Print completion message
echo "Setup complete. Activate your environment with 'conda activate carla_env' and start CARLA with './CARLA_0.9.11/CarlaUE4.sh -opengl -RenderOffScreen'."
EOF

echo "Switch to the new user and run CARLA:"
echo "su - $USERNAME"
