### Train
1. Pretrain weights:

    Download the pretrained segformer here [pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing).

2. Config

    Edit config file in `configs.py`, including dataset and network settings.

3. Run multi GPU distributed training:
    ```shell
    $ CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py
    ```

- The tensorboard file is saved in `log_<datasetName>_<backboneSize>/tb/` directory.
- Checkpoints are stored in `log_<datasetName>_<backboneSize>/checkpoints/` directory.

### Evaluation
1. Download checkpoints:
   
    https://o365inha-my.sharepoint.com/:f:/g/personal/sychoi_office_inha_ac_kr/EoQMB96V5KtMtugaUfQ4GXgBCwbvZH0lCp2jy6cGbDxqPg?e=C7lxTO

2. Run the evaluation by:
    ```shell
    CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -d="Device ID" -e="epoch number or range"
    ```
    If you want to use multi GPUs please specify multiple Device IDs (0,1,2...).





## Acknowledgement
Our code is based on [CMX] (https://github.com/huaaaliu/RGBX_Semantic_Segmentation.git), thanks for their excellent work :)
