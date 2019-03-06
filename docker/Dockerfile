FROM docker.io/paperspace/tensorflow-python:latest

RUN pip3 install \
	keras \
	tqdm \
	opencv-python

# need to add a few other packages for opencv to work
RUN apt-get update && apt-get install -y libsm6 libxext6 && apt-get install -y libxrender-dev