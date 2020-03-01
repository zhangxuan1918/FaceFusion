FROM xuan1918/tensorflow/3dmm-rendering:v0.1.1-py3

ARG SSH_PRV_KEY
ARG SSH_PUB_KEY
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev python3.7-dev

# Add the keys and set permissions
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

RUN echo "$SSH_PRV_KEY" > /root/.ssh/id_rsa && \
    echo "$SSH_PUB_KEY" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

RUN pip install --upgrade pip

# install tensorflow3DMMRendering from github
RUN cd ~ && git clone git@github.com:zhangxuan1918/tensorflow3DMMRendering.git && \
 	python -m pip install tensorflow3DMMRendering/

RUN rm -rf /root/.ssh/

# install tensorflow models
RUN git clone https://github.com/zhangxuan1918/models.git /tensorflow_models
RUN pip install -r /tensorflow_models/official/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/src"
ENV PYTHONPATH "${PYTHONPATH}:/tensorflow_models/"

# install requirements
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt
