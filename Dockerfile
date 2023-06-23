FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/hu.archive.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update 

RUN apt-get install -y python3 python3-pip git wget espeak
RUN pip3 install --upgrade pip

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION=v1.13.1

RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /opt
RUN git clone https://github.com/legekka/GanyuTTS -b docker

WORKDIR /opt/GanyuTTS
RUN pip3 install -r requirements.txt

RUN mkdir -p /opt/GanyuTTS/models
RUN mkdir -p /opt/GanyuTTS/tmp
RUN wget https://legekka.fs.boltz.hu/mv3cna.pth -O /opt/GanyuTTS/models/pretrained_vctk.pth
RUN wget https://legekka.fs.boltz.hu/op4olp.pth -O /opt/GanyuTTS/models/ganyu_27+14.pth
RUN wget https://legekka.fs.boltz.hu/c2tnk4.pt -O /opt/GanyuTTS/hubert/checkpoint_best_legacy_500.pt

EXPOSE 4111:4111

ENTRYPOINT [ "python3", "api.py" ]