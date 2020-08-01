FROM xuan1918/tensorflow/3dmm-rendering:v0.2.3-py3

ARG GITHUB_PAT

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev python3.7-dev \
    libglib2.0-0 libgtk-3-dev libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev

RUN pip install --upgrade pip

# install tensorflow3DMMRendering
RUN pip install git+https://zhangxuan1918:$GITHUB_PAT@github.com/zhangxuan1918/tensorflow3DMMRendering.git

# install tensorflow models
#RUN git clone https://github.com/zhangxuan1918/models.git /tensorflow_models
#RUN pip install -r /tensorflow_models/official/requirements.txt
#ENV PYTHONPATH "${PYTHONPATH}:/tensorflow_models/"

# install requirements
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# add CUPTI to library
#ENV LD_LIBRARY_PATH "/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}" # not working
RUN echo '/usr/local/cuda/extras/CUPTI/lib64' >> /etc/ld.so.conf
RUN ldconfig && ldconfig -p | grep libcupti