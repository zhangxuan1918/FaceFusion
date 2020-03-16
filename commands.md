## build docker container
docker-compose build --build-arg GITHUB_PAT="$(cat ~/.ssh/github_xuan_desktop_token)"

## Tensorboard profiler

1. start training in pycharm, e.g. run `train_3dmm_unsupervised.py`
    * the code set up server at port 6019
    
    ```python 
    profiler.start_profiler_server(6019)
    ```
2. run `nvidia-docker exec -it face_fusion tensorboard --logdir /opt/data/face-fuse/model/20200310/unsupervised/summaries/profiler --host 0.0.0.0 --port 6009`
3. in google chrome, open `localhost:6009`
4. on the right top of the tab, select `PROFILE`
5. click `Capture Profile`
6. the URL is `localhost:6019`