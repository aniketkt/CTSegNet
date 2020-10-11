# An alternative method is starting from a tensorflow image and skip to directly installing CTSegNet with pip3
FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

MAINTAINER Aniket Tekawade <aniketkt@gmail.com>

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}


COPY . CTSegNet/.


RUN	apt-get update && \
	apt install libgl1-mesa-glx -y


RUN	pip install --upgrade pip && \
	pip install CTSegNet/. && \
	pip install git+https://github.com/aniketkt/ImageStackPy.git#egg=ImageStackPy
	
