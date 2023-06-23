FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/hu.archive.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update 

RUN apt-get install -y python3 python3-pip git wget
RUN pip3 install --upgrade pip

# we want to clone the repo and install the requirements in the /opt directory
WORKDIR /opt
RUN git clone https://github.com/legekka/GanyuTTS -b docker

RUN mkdir -p /opt/GanyuTTS/models
WORKDIR /opt/GanyuTTS
RUN wget https://legekka.fs.boltz.hu/mv3cna.pth -O models/pretrained_vctk.pth
RUN wget https://legekka.fs.boltz.hu/op4olp.pth -O models/ganyu_27+14.pth

# install pytorch
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION=v1.13.1

RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# install the requirements
WORKDIR /opt/GanyuTTS
RUN pip3 install -r requirements.txt


ENTRYPOINT [ "python3", "api.py" ]