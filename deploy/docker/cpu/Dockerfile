FROM ubuntu:14.04
MAINTAINER Dmitry Senin <d.senin@samsung.com>

# Change this variables if something doesn't match your configuration
ENV SMAUG_IP_ADDR 106.109.131.111

# Install Veles
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget
RUN echo "deb http://${SMAUG_IP_ADDR}/apt trusty main" >> /etc/apt/sources.list && \
    wget -O - http://${SMAUG_IP_ADDR}/apt/smaug.rnd.samsung.ru.gpg.key 2> /dev/null | apt-key add - && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-veles

RUN apt-get clean

ENTRYPOINT ["veles", "-p", "''", "--disable-opencl"]
CMD ["-s", "/usr/lib/python3/dist-packages/veles/znicz/samples/wine.py", "-"]
