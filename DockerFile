# use cuda 11.6 as the base image
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    # python3.7 \
    # python3-pip \
    bzip2 \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# update alternatives to set python 3.7 as the default python
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# install anaconda by executing the installed shell script and installing into a directory
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh 
RUN mkdir /opt/anaconda3
RUN echo bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p /opt/anaconda3
RUN rm Anaconda3-2024.06-1-Linux-x86_64.sh

# set environment variables for anaconda
ENV CONDA_PATH /opt/anaconda3/bin:$PATH

# create and activate conda environment
# RUN conda create -y -n carla_env python=3.7 
# RUN echo "source activate carla_env" > ~/.bashrc
# SHELL ["/bin/bash", "-c", "source ~/.bashrc"]

# use a build argument to invalidate the cache for the git clone step
# ARG CACHEBUST=1

# clone the MAPDDPG repository
WORKDIR /opt
RUN git clone https://github.com/ENDEAVR-E2E-Autonomous-Driving/MAPDDPG-CARLA

# cd the working directory
WORKDIR /opt/MAPDDPG-CARLA

# install python dependencies
# RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN /opt/anaconda3/envs/carla_env/bin/pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN /opt/anaconda3/envs/carla_env/bin/pip install -r requirements.txt

# download and install carla
WORKDIR /opt
# RUN apt-get install libomp5 -y
RUN apt-get update && apt-get install -y libomp-11-dev
RUN wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
RUN mkdir /opt/carla-simulator
RUN tar -xzvf CARLA_0.9.15.tar.gz -C /opt/carla-simulator/
# RUN pip install /carla-simulator/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.whl

# set environment variables for CARLA
ENV CARLA_ROOT /opt/carla-simulator
ENV PATH $CARLA_ROOT/bin:$PATH 

# change to MAPDDPG directory
WORKDIR /opt/MAPDDPG-CARLA

# command to run carla in off-screen mode and start the training script
# CMD ["bash", "-c", "source activate carla_env && $CARLA_ROOT/CarlaUE4.sh -opengl -RenderOffScreen & python main.py"]

# Entry point
CMD ["/bin/bash"]
