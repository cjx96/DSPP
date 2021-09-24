DSPP
===

The source code is for the paper: “Deep Structural Point Process for Learning Temporal Interaction Networks” accepted in ECML/PKDD 2021 by Jiangxia Cao, Xixun Lin, Xin Cong, Shu Guo, Hengzhu Tang, Tingwen Liu, Bin Wang.


```
@inproceedings{cao2021dspp,
  title={Deep Structural Point Process for Learning Temporal Interaction Networks},
  author={Cao, Jiangxia and Lin, Xixun and Cong, Xin and Guo, Shu and Tang, Hengzhu and Liu, Tingwen and Wang, Bin},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML/PKDD)},
  year={2021}
}
```

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Preparation
---

We provide a demo dataset, and the other datasets can be downloaded from the [Jodie](https://github.com/srijankr/jodie). 

Note that Last.FM, Wikipedia and Reddit are required downloaded in the `./dataset` directory.

Usage
---

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU, 256 RAM and a Tesla V100 32GB GPU. Note that we first preprocess the dataset by t-batch manner, and then run DSPP to make prediction.

Demo:

```shell
CUDA_VISIBLE_DEVICES=0 python -u train.py --id debug --network lastfmdebug --undebug
```

Preprocessing and running example:

```shell
cd dataset
python data_preprocess.py --network reddit --graphsnapshot 1024 --sequence_length 40

cd ..
CUDA_VISIBLE_DEVICES=0 python -u train.py --id reddit_1024_40 --network reddit_1024_40 --undebug
```

