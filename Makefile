docker-build:
	sudo docker build -t captucha_identifier -f Dockerfile .

docker-run:
	if [ ! -d ./results ]; then mkdir ./results; fi
	sudo docker run -it --rm \
		--mount type=bind,source="$(shell pwd)"/models,target=/my_project/models \
		--mount type=bind,source="$(shell pwd)"/results,target=/my_project/results \
		captucha_identifier
