FROM ubuntu:16.04

RUN echo "install python" && \
	apt-get -y update && \
	apt-get -y install python3 && \
	apt-get -y install curl && \
	apt-get -y install git && \
	curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
	/usr/bin/python3 get-pip.py && \
	rm get-pip.py && \
	pip install jupyter nengo matplotlib nengo-loihi && \
	mkdir /Notebooks
	
EXPOSE 8888
WORKDIR /Notebooks
ENTRYPOINT jupyter notebook --no-browser --allow-root --port 8888 --ip 0.0.0.0



