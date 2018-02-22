FROM daocloud.io/jermine/pytorch
MAINTAINER Jermine.hu@qq.com
WORKDIR /app
COPY . /app
RUN pip3 install -r /app/requirements.txt  -i  https://mirrors.aliyun.com/pypi/simple
EXPOSE 8080
EXPOSE 5000
CMD [ "/app/run.sh" ]