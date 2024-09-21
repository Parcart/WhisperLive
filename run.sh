docker stop whisperlive
docker rm -f whisperlive
docker build -t whispergpulive .
docker image prune -f
docker run -it --gpus all -p 9090:9090 --name whisperlive whispergpulive:latest
