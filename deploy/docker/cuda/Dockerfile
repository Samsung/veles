FROM ubuntu:14.04
MAINTAINER Dmitry Senin <d.senin@samsung.com>

# Change this variables if something doesn't match your configuration
ENV CUDA_RUN cuda_6.5.14_linux_64.run
ENV CUDA_RUN_HOST_PATH $CUDA_RUN
ENV CUDA_RUN_DOCKER_PATH /data/packages/bin/
ENV EXTRACT_PATH /tmp/nvidia/extracted/
ENV SMAUG_IP_ADDR 106.109.131.111

# Install NVIDIA Driver and CUDA Toolkit
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential linux-headers-$(uname -r) wget && \
    DEBIAN_FRONTEND=noninteractive apt-get --purge remove -y nvidia*
COPY $CUDA_RUN_HOST_PATH $CUDA_RUN_DOCKER_PATH
RUN chmod +x $CUDA_RUN_DOCKER_PATH/$CUDA_RUN && \
    mkdir -p $EXTRACT_PATH
RUN $CUDA_RUN_DOCKER_PATH/$CUDA_RUN -extract=$EXTRACT_PATH && \
    $EXTRACT_PATH/NVIDIA-Linux-x86_64-*.run -s -N --no-kernel-module && \
    $EXTRACT_PATH/cuda-linux64-rel-*.run -noprompt && \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc

# Install Veles
RUN echo "deb http://${SMAUG_IP_ADDR}/apt trusty main" >> /etc/apt/sources.list && \
    wget -O - http://${SMAUG_IP_ADDR}/apt/smaug.rnd.samsung.ru.gpg.key 2> /dev/null | apt-key add - && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-veles

RUN apt-get clean && rm -rf $EXTRACT_PATH $CUDA_RUN_DOCKER_PATH/$CUDA_RUN

ENTRYPOINT ["veles", "-p", "''"]
CMD ["-s", "/usr/lib/python3/dist-packages/veles/znicz/samples/wine.py", "-"]
