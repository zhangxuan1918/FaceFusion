FROM xuan1918/tensorflow/3dmm-rendering:v0.1.1-py3

ARG SSH_PRV_KEY
ARG SSH_PUB_KEY

# Add the keys and set permissions
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

RUN echo "$SSH_PRV_KEY" > /root/.ssh/id_rsa && \
    echo "$SSH_PUB_KEY" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

# install tensorflow3DMMRendering from github
RUN cd ~ && git clone git@github.com:zhangxuan1918/tensorflow3DMMRendering.git && \
 	python -m pip install tensorflow3DMMRendering/

RUN rm -rf /root/.ssh/

# install other requirement
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt