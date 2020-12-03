# Docker Install Instructions

Before starting make sure that docker is installed and the docker daemon is running.
After the installation you will usually have to add your user to the "docker" group in order to be able to execute commands (or use sudo).
For ubuntu:
```
apt install docker.io docker
sudo usermod -a -G docker $YOURUSERNAME`			<-- careful here, do NOT forget -a
```
at this point you will have to log in again to be added to the group, alternatively 
'newgrp docker' might do the trick.

## Building the docker image

In the directory with the Dockerfile simply run:

docker build .

You can optionally provide a tag so you can easily identify your image. For example:

docker build -t loihi .

Docker images are the 'blue prints' for containers. To create a new container from the image you just created you can use:

docker run -p 8888:8888 loihi

The -p option is for binding ports within the docker container to ports on the host machine. We need this to be able to access the notebook server running in the container.
At this point your terminal should show the familiar jupyter notebook start prints. You should be able to access the notebook by copying the url and token into your browser.

## Copying files

The docker image has git installed, so you can just clone the git repository into the container. You can connect to you container and open a shell like this:

1. find out the id of your docker container:
	docker container ls
2. Attach your container and open a shell
	docker exec -it $CONTAINER_ID /bin/bash
3. The notebook server is running in folder /Notebooks. If you clone the repo there you will be able to access it in your browser.
Alternatively copy files to the container using dpcker cp. For example:
docker cp test.py $CONTAINER_ID:/Notebooks

## Stopping containers

1. list running containers
	docker container ls
2. stop with
	docker stop $CONTAINER_ID

Make sure to clean up containers since they take up quite some memory.

## Restarting a stopped container

1. docker start $CONTAINER_ID, you can find a list of all containers, including stopped ones using 'docker container ls -a'

## A word of caution (totally not from experience)

If you delete a (stopped) docker container everything in the container is GONE. Make sure you save your work before deleting containers. Stopping and restarting a container should (?) totally be fine, but if you create a new container using docker run it will make a new container based on your image without any of your previous work.
