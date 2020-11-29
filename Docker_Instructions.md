# Docker Install Instructions

Before starting make sure that docker is installed and the docker daemon is running.
After the installation you will usually have to add your user to the "docker" group in order to be able to execute commands (or use sudo).
For ubuntu:

apt install docker.io docker
sudo usermod -a -G docker $YOURUSERNAME			<-- careful here, do NOT forget -a

at this point you will have to log in again to be added to the group, alternatively 
'newgrp docker' might do the trick

## Building the docker image

In the directory with the docker image simply run:

docker build .

You can optionally provide a tag so you can easily identify your image. For example:

docker build -t loihi .

Docker images are the 'blue prints' for containers. To create a new container from the image you just created you can use:

docker run -p 8888:8888 loihi

The -p option is for binding ports within the docker container to ports on the host machine. We need this to be able to access the notebook server running in the container.
At this point your terminal should show the familiar jupyter notebook start prints. You should be able to access the notebook by copying the url and token into your browser.
