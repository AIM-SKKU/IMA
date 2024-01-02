# Train
1. Config
  Edit config file in ./configs.

    i. Set inference.gpu_id = CUDA_VISIBLE_DEVICES.
    E.g.,inference.gpu_id = "0,1,2,3".

    ii. Set dataset_root = path_to_dataset.
     E.g.,dataset_root = "/root/datasets/ShapeConv/nyu_v2".
     
2. Run
    i. Distributed training
    ```bash
    ./tools/dist_train.sh config_path gpu_num
    ```
    E.g., train shape-conv model on NYU-V2(40 categories) with 4 GPUs, please run:
    ```bash
    ./tools/dist_train.sh configs/nyu/nyu40_deeplabv3plus_resnext101_shape.py 4
    ```

# Evaluation
1. Download checkpoints:
   
    https://o365inha-my.sharepoint.com/:f:/g/personal/sychoi_office_inha_ac_kr/EoQMB96V5KtMtugaUfQ4GXgBCwbvZH0lCp2jy6cGbDxqPg?e=C7lxTO
4. Run
    i. Distributed evaluation, please run:
    ```bash
    ./tools/dist_test.sh config_path checkpoint_path gpu_num
    ```
    - `config_path` is path of config file;
    - `checkpoint_path` is path of model file;
    - `gpu_num` is the number of GPUs used, note that `gpu_num <= len(infernece.gpu_id).`


    E.g., evaluate shape-conv model on NYU-V2(40 categories), please run:
    ```bash
    ./tools/dist_test.sh configs/nyu/nyu40_deeplabv3plus_resnext101_shape.py checkpoints/nyu40_deeplabv3plus_resnext101_shape.pth 4
    
