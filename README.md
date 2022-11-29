# BayeSeg
The official implementation of "[Joint Modeling of Image and Label Statistics for Enhancing Model Generalizability of Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_35)", which has been accepted by MICCAI 2022.

## Content
- [BayeSeg](#bayeseg)
  - [Content](#content)
  - [Dependencies](#dependencies)
  - [Quick test](#quick-test)
  - [How to train](#how-to-train)
  - [Citation](#citation)

## Dependencies
BayeSeg was implemented on *Ubuntu 16.04* with *Python 3.6*. Before training and test, please create an environment via [Anaconda](https://www.anaconda.com/) (suppose it has been installed on your computer), and install pytorch 1.10.2, as follows,
```bash
conda create -n BayeSeg python=3.6
source activate BayeSeg
conda install torch==1.10.2
```
Besides, please install other packages using ```pip install -r requirements.txt```.

## Quick test
BayeSeg was tested on the public datasets from [MICCAI 2017 ACDC](https://acdc.creatis.insa-lyon.fr/#) and [MICCAI 2019 MS-CMRSeg](https://zmiclab.github.io/zxh/0/mscmrseg19/). For ACDC, all training cases were used for test. For MS-CMRSeg, 15 cases were randomly selected for test.

|Datasets/Models|Parameters|BaiduPan|OneDrive|
|-----|-------|--------|--------|
|ACDC |  -     |[link](https://acdc.creatis.insa-lyon.fr/#)|[link](https://acdc.creatis.insa-lyon.fr/#)|
|MS-CMRSeg|-   |[link](https://pan.baidu.com/s/1MlrRxYhmp9CRabgn0AeFog) `s4t8`|[link](https://1drv.ms/u/s!AuJaQmQJN4arhGcbso7reViO9rF1?e=a3YCn2)|
|Unet   |25.8M |[link](https://pan.baidu.com/s/1LM0GeP80QO73hqEHbZQxqg) `1zgr`|[link](https://1drv.ms/u/s!AuJaQmQJN4arhGtwFDVlfny5HbrS?e=vPk51I)|
|PUnet  |5.0M  |[link](https://pan.baidu.com/s/1mCzlgRHdfCCsBYuEy1fleg) `07rm`|[link](https://1drv.ms/u/s!AuJaQmQJN4arhGgzZZAmMclhx9mi?e=tt6nmr)|
|Baseline|26.9M|[link](https://pan.baidu.com/s/1IuBEnsiLAnmqOJqst64vrQ) `1i7y`|[link](https://1drv.ms/u/s!AuJaQmQJN4arhGk1KUA6DlCSzfnJ?e=l8Wwkx)|
|BayeSeg|26.9M |[link](https://pan.baidu.com/s/1C3EqfR3fFnF0D0pfTMoiwA) `0an5`|[link](https://1drv.ms/u/s!AuJaQmQJN4arhGr7ty9owsE9gCEK?e=P8qjy6)|

- `ACDC` comes from MICCAI 2017 ACDC, one needs to download it from its official homepage.
- `MS-CMRSeg.zip` contains three folders, i.e., train, val, and test. 
  - `train` contains 25 subjects randomly selected from LGE CMR of MS-CMRSeg
  - `val` contains 5 subjects randomly selected from LGE CMR of MS-CMRSeg
  - `test` contains three sequences, i.e., C0 (bSSFP CMR), LGR (LGE CMR), and T2 (T2-weighted CMR), and each sequence consists of 15 subjects randomly selected from MS-CMRSeg. 
- `Unet.zip` contains the checkpoint of U-Net model, which was trained on LGE CMR using cross-entropy.
- `PUnet.zip` contains the checkpoint of PU-Net model, which was trained on LGE CMR using its default loss.
- `Baseline.zip` contains the checkpoint of Baseline model, which was trained on LGE CMR only using cross-entropy.
- `BayeSeg.zip` contains the checkpoint of BayeSeg model, which was trained on LGE CMR using an additional variational loss.

We have provided the script of testing U-Net, PU-Net, Baseline, and BayeSeg in `demo.sh`. Please start testing these models as follows.

The setting of test directory is defined in `inference.py` as follows,
```python
if dataset in ['MSCMR', 'ACDC']:
    test_folder = "../Datasets/{}/test/{}/images/".format(dataset, sequence)
    label_folder = "../Datasets/{}/test/{}/labels/".format(dataset, sequence)
else:
    raise ValueError('Invalid dataset: {}'.format(dataset))
```
For ACDC, one need to download this dataset from its homepage, and then prepare test data as above.

To test the performance of U-Net, PU-Net, Baseline, and BayeSeg on the LGE CMR of MS-CMRSeg, please uncomment the corresponding line in `demo.sh`, and then run `sh demo.sh`.
```bash
# test Unet
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model Unet --eval --dataset MSCMR --sequence LGR --resume logs/Unet/checkpoint.pth --output_dir results --device cuda

# test PUnet
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model PUnet --eval --dataset MSCMR --sequence LGR --resume logs/PUnet/checkpoint.pth --output_dir results --device cuda

# test baseline
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model Baseline --eval --dataset MSCMR --sequence LGR --resume logs/Baseline/checkpoint.pth --output_dir results --device cuda

# test BayeSeg
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/BayeSeg/checkpoint.pth --output_dir results --device cuda
```
Here, `--sequence` can be set to C0, LGR, or T2 for MS-CMRSeg, and C0 for ACDC. For example, to test the cross-sequence segmentation performance of U-Net, PU-Net, Baseline, and BayeSeg on the T2-weighted CMR of MS-CMRSeg, please set `--sequence LGR` to `--sequence T2`.

## How to train
All models were trained using LGE CMR of MS-CMRSeg, and the root of training data is defined in `data/mscmr.py` as follows,
```python
root = Path('your/dataset/directory' + args.dataset)
```
Please replace `your/dataset/directory` with your own directory.

To train U-Net, PU-Net, Baseline, and BayeSeg, please uncomment the corresponding line in `demo.sh`, and run `sh demo.sh`.
```bash
# train Unet
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model Unet --batch_size 8 --output_dir logs/Unet --device cuda

# train PUnet
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model PUnet --batch_size 8 --output_dir logs/PUnet --device cuda

# train Baseline
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model Baseline --batch_size 8 --output_dir logs/Baseline --device cuda

# train BayeSeg
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model BayeSeg --batch_size 8 --output_dir logs/BayeSeg --device cuda
```

## Citation
If our work is helpful in your research, please cite this as follows.

[1] S. Gao, H. Zhou, Y. Gao, and X. Zhuang, "Joint Modeling of Image and Label Statistics for Enhancing Model Generalizability of Medical Image Segmentation," MICCAI 2022, pp. 360-369. [[arXiv]](https://arxiv.org/abs/2206.04336) [[MICCAI]](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_35)
```
@InProceedings{Gao/BayeSeg/2022,
  author = {Gao, Shangqi and Zhou, Hangqi and Gao, Yibo and Zhuang, Xiahai},
  title = {Joint Modeling of Image and Label Statistics for Enhancing Model Generalizability of Medical Image Segmentation},
  booktile = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  year = 2022,
  publisher = {Springer Nature Switzerland},
  pages = {360--369}
}
```

Don't hesitate to contact us via [shqgao@163.com]() or [zxh@fudan.edu.cn](), if you have any questions.
