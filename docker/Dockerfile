FROM docker.io/paperspace/tensorflow-python:latest

RUN pip3 install \
	keras \
	tqdm \
	opencv-python \
	Pillow \
	shapely \
	imgaug

# need to add a few other packages
RUN apt-get update && apt-get install -y \
		libsm6 \
		libxext6 \
		python3-tk &&\
	apt-get install -y libxrender-dev