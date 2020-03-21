from project_code.training.train_3dmm_supervised import TrainFaceModelSupervised

train_model = TrainFaceModelSupervised(
        bfm_dir='/opt/data/BFM/',
        n_tex_para=40,  # number of texture params used
        data_dir='/opt/data/face-fuse/',  # data directory for training and evaluating
        model_dir='/opt/data/face-fuse/model/test/supervised/',  # model directory for saving trained model
        epochs=10,  # number of epochs for training
        train_batch_size=64,  # batch size for training
        eval_batch_size=64,  # batch size for evaluating
        steps_per_loop=10,  # steps per loop, for efficiency
        initial_lr=0.00005,  # initial learning rate
        init_checkpoint=None,  # initial checkpoint to restore model if provided
        init_model_weight_path='/opt/data/face-fuse/model/face_vgg_v2/weights.h5',
        # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
        resolution=224,  # image resolution
        num_gpu=1,  # number of gpus
        stage='UNSUPERVISED',  # stage name
        backbone='resnet50',  # model architecture
        distribute_strategy='mirror',  # distribution strategy when num_gpu > 1
        run_eagerly=True,
        model_output_size=290,
        enable_profiler=False
    )
train_model.init_model()
train_model.init_bfm()
train_model.create_evaluating_dataset()
train_model._run_evaluation(1, train_model.eval_dataset)