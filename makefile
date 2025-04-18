# Makefile

IMAGE = ghcr.io/cinnamon/kotaemon:main-full
PLATFORM = linux/arm64
PORT = 7860
HOST_PORT = 7860
VOLUME = ./ktem_app_data:/app/ktem_app_data

run:
	docker run \
	-e GRADIO_SERVER_NAME=0.0.0.0 \
	-e GRADIO_SERVER_PORT=$(PORT) \
	-v $(VOLUME) \
	-p $(HOST_PORT):$(PORT) \
	-it --rm \
	--platform $(PLATFORM) \
	$(IMAGE)
