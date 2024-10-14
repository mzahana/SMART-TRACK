.PHONY: px4-dev-simulation-ubuntu22 px4-simulation-cuda11.7-ubuntu22

all: px4-dev-simulation-ubuntu22 px4-simulation-cuda11.7-ubuntu22


px4-dev-simulation-ubuntu22:
	docker build -t mzahana/px4-dev-simulation-ubuntu22 . -f Dockerfile_simulation-ubuntu22
px4-simulation-cuda11.7.1-ubuntu22:
	docker build -t mzahana/px4-simulation-cuda11.7.1-ubuntu22 . -f Dockerfile_simulation-cuda11.7.1-ubuntu22
px4-simulation-cuda12.2.0-ubuntu22:
	docker build -t mzahana/px4-simulation-cuda12.2.0-ubuntu22 . -f Dockerfile_simulation-cuda12.2.0-ubuntu22
px4.simulation.ubuntu22.wsl:
	docker build -t mzahana/px4.simulation.ubuntu22.wsl . -f Dockerfile.simulation.ubuntu22.wsl

