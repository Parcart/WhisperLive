docker stop WhisperLiveGPU
docker stop WhisperLiveCPU
docker rm -f WhisperLiveGPU
docker rm -f WhisperLiveCPU
docker build -t whispergpulive .
docker image prune -f
docker container create -it --gpus all -p 9090:9090 --name WhisperLiveGPU whispergpulive:latest
docker container create -it -p 9090:9090 --name WhisperLiveCPU whispergpulive:latest
