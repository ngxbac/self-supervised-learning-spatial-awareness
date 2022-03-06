APP_NAME=ngxbac/ss:20.4.1
CONTAINER_NAME=ss_20.4.1

run: ## Run container
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME} \
		-v $(shell pwd)/data/:/data \
		-v $(shell pwd)/logs/:/logs \
		-v $(shell pwd):/code/ $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}

bratsseg:
	bash bin/brats_segment.sh

bratstx:
	bash bin/brats_temporalmix.sh

structsegseg:
	bash bin/structseg_segment.sh

structsegtx:
	bash bin/structseg_temporalmix.sh