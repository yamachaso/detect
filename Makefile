force_build=true

init:
ifeq ($(force_build), true)
	docker build -f Dockerfile.base -t yamachaso/cuda_detectron2_ros:latest .
else
	docker pull yamachaso/cuda_detectron2_ros:latest
endif

	cp .devcontainer/devcontainer_example.json .devcontainer/devcontainer.json
	docker compose build

start:
	docker compose up -d --force-recreate

stop:
	docker compose stop

detectron2:
	docker compose exec detectron2 bash
