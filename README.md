# Unbiased Directed Object Attention Graph for Object Navigation
Ronghao Dang, Zhuofan Shi, Liuyi Wang, Zongtao He, Chengju Liu, Qijun Chen (Under review for ACMMM 2022)

[Arxiv Paper](https://arxiv.org/abs/2204.04421)

<p align="center"><img src="fig/model_architecture.png" width="700" /></p>

## Abstract
We explore the object attention bias problem in object navigation task. Therefore, we propose the DOA graph and novel cross-attention method to solve the problem. Our overall model achieves a SOTA level.
## Setup
- Clone the repository `git clone http://github.com/gold-d/DOA.git` and move into the top level directory `cd DOA`
- Create conda environment. `pip install -r requirements.txt`
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). The offline data is discretized from [AI2-Thor](https://ai2thor.allenai.org/) simulator.
- Download the [pretrain dataset](https://drive.google.com/file/d/1dFQV10i4IixaSUxN2Dtc6EGEayr661ce/view), which refers to [VTNet](https://github.com/xiaobaishu0097/ICLR_VTNet).
The `data` folder should look like this
```python
data/ 
    └── Scene_Data/
        ├── FloorPlan1/
        │   ├── resnet18_featuremap.hdf5
        │   ├── graph.json
        │   ├── visible_object_map_1.5.json
        │   ├── det_feature_categories.hdf5
        │   ├── grid.json
        │   └── optimal_action.json
        ├── FloorPlan2/
        └── ...
    └── AI2Thor_VisTrans_Pretrain_Data/
        ├── data/
        ├── annotation_train.json
        ├── annotation_val.json
        └── annotation_test.json
``` 
## Training and Evaluation

### Pre-train our DOA model

`python main_pretraining.py --title DOA_Pretrain --model DOA_Pretrain --workers 9 --gpu-ids 0 --epochs 20 --log-dir runs/pretrain --save-model-dir trained_models/pretrain`
### Train our DOA model
`python main.py --title DOA --model DOA --workers 9 --gpu-ids 0 --max-ep 3000000 --log-dir runs/RL_train --save-model-dir trained_models/RL_train --pretrained-trans trained_models/pretrain/checkpoint0004.pth` 
### Evaluate our DOA model
`python full_eval.py --title DOA --model DOA --results-json eval_best_results/DOA.json --gpu-ids 0 --log-dir runs/RL_train --save-model-dir trained_models/RL_train`  
## Citing
If you find this project useful in your research, please consider citing:
```
@article{dang2022unbiased,
  title={Unbiased Directed Object Attention Graph for Object Navigation},
  author={Dang, Ronghao and Shi, Zhuofan and Wang, Liuyi and He, Zongtao and Liu, Chengju and Chen, Qijun},
  journal={arXiv preprint arXiv:2204.04421},
  year={2022}
}
```
