FROM xuan1918/tensorflow/3dmm-rendering:v0.1.0-py3

# install requirement
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt