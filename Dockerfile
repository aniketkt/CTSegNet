#FROM	tensorflow/tensorflow:latest-gpu-jupyter
FROM	tensorflow/tensorflow:2.3.0-gpu-jupyter
#FROM	nvidia/cuda:10.1-base
# install git

#RUN	apt-get update -y
#RUN	apt-get upgrade -y
#RUN	apt-get install git -y

#RUN	git clone https://github.com/aniketkt/CTSegNet.git
COPY	. CTSegNet/.
RUN	pip install CTSegNet/.

