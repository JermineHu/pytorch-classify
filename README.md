# The pytorch demo

## Docker build

```
docker build -t pytorch_demo .
```

## Run a instance by docker 

```
docker run -it --rm -e API_PORT=8080 -e MODEL_PATH="http://192.168.16.189:1002/resNet.pth" --e DOCS_SERVER=http://192.168.16.254:8082/download/docs/?url=http://192.168.16.254:8080/assets/docs/api.yml --name pytorch_demo  pytorch_demo

```

#### The REST Ful API was running on http://192.168.16.189:8080
#### The REST Ful API Document was running on http://192.168.16.254:8082/download/docs/?url=http://192.168.16.254:8080/assets/docs/api.yml