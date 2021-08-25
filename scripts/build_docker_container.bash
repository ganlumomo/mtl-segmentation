container_name=$1

host +local:
docker run -it --net=host --runtime=nvidia \
	--user=$(id -u) \
	--shm-size 8G \
	-e DISPLAY=$DISPLAY \
	-e QT_GRAPHICSSYSTEM=native \
	-e CONTAINER_NAME=cuda \
	-e USER=$USER \
	--workdir=/home/$USER \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	-v "/etc/group:/etc/group:ro" \
	-v "/etc/passwd:/etc/passwd:ro" \
	-v "/etc/shadow:/etc/shadow:ro" \
	-v "/etc/sudoers.d:/etc/sudoers.d:ro" \
	-v "/home/$USER/code/DockerFolder/:/home/$USER/" \
	-v "/media/sde1/cel/data/:/home/$USER/data/" \
	--device=/dev/dri:/dev/dri \
	--name=${container_name} \
	umcurly/nvidia-segmgentation:latest
