docker stop WhisperLiveGPUNSM
docker stop WhisperLiveCPUNSM
docker stop WhisperLiveGPU
docker stop WhisperLiveCPU
docker rm -f WhisperLiveGPUNSM
docker rm -f WhisperLiveCPUNSM
docker rm -f WhisperLiveGPU
docker rm -f WhisperLiveCPU
docker build -t whispergpulive .
docker image prune -f
docker container create -e NSM_FLAG="-nsm" -it --gpus all -p 9090:9090 --name WhisperLiveGPUNSM whispergpulive:latest
docker container create -e NSM_FLAG="-nsm" -it -p 9090:9090 --name WhisperLiveCPUNSM whispergpulive:latest
docker container create -it --gpus all -p 9090:9090 --name WhisperLiveGPU whispergpulive:latest
docker container create -it -p 9090:9090 --name WhisperLiveCPU whispergpulive:latest
