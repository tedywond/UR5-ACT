SHELL := /bin/bash
# .PHONY: all clean build push build_sing_from_daemon build_sing

PROJ_NAME := octo
TAG := latest
DOCKER_NAME := ripl/$(PROJ_NAME):$(TAG)
SINGULARITY_NAME := $(PROJ_NAME)-$(TAG).sif
DOCKER := $(shell type -p nvidia-docker || echo docker)

# HACK to use colon (https://stackoverflow.com/a/61016478/19913466)
colon := :
$(colon) := :

default: build

build:
	docker build -t $(DOCKER_NAME) .

clean:
	docker build --no-cache -t $(DOCKER_NAME) .

push:
	docker push $(DOCKER_NAME)

build_sing_from_daemon:
	singularity build $(SINGULARITY_NAME) docker-daemon$(:)//$(DOCKER_NAME)
	rsync -vz $(SINGULARITY_NAME) takuma@slurm.ttic.edu:/share/data/ripl-takuma/singularity-workspace/$(PROJ_NAME)/

build_sing: build build_sing_from_daemon
