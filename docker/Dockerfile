# An alternative method is starting from a tensorflow image and skip to directly installing CTSegNet with pip3
FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

MAINTAINER Aniket Tekawade <aniketkt@gmail.com>

COPY . CTSegNet/.


RUN	apt-get update && \
	apt install libgl1-mesa-glx -y


RUN	pip install --upgrade pip && pip install CTSegNet/.
