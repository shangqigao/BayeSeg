# BayeSeg (is buiding)
The official implementation of "[Joint Modeling of Image and Label Statistics for Enhancing Model Generalizability of Medical Image Segmentation](https://arxiv.org/abs/2206.04336)", which has been accepted by MICCAI 2022.

## Content
- [BayeSeg (is buiding)](#bayeseg-is-buiding)
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
|ACDC |       |[link](https://acdc.creatis.insa-lyon.fr/#)|[link](https://acdc.creatis.insa-lyon.fr/#)|
|MS-CMRSeg|   |[link]()|[link]()|
|Unet   |     |[link]()|[link]()|
|PUnet  |     |[link]()|[link]()|
|Baseline|    |[link]()|[link]()|
|BayeSeg|     |[link]()|[link]()|

- `ACDC` comes from MICCAI 2017 ACDC, one needs to download it from its official homepage.
- `MS-CMRSeg.zip` contains three sequences, i.e., C0 (bSSFP CMR), LGR (LGE CMR), and T2 (T2-weighted CMR), and each sequence consists of train, val, and test datasets. For example, LGR is comprised of the following three datasets:
  - `train` contains 25 subjects randomly selected from LGE CMR of MS-CMRSeg
  - `val` contains 5 subjects randomly selected from LGE CMR of MS-CMRSeg
  - `test` contains 15 subjects randomly selected from LGE CMR of MS-CMRSeg
- `Unet.zip` contains the checkpoint of U-Net model, which was trained on LGE CMR using cross-entropy.
- `PUnet.zip` contains the checkpoint of PU-Net model, which was trained on LGE CMR using its default loss.
- `Baseline.zip` contains the checkpoint of Baseline model, which was trained on LGE CMR only using cross-entropy.
- `BayeSeg.zip` contains the checkpoint of BayeSeg model, which was trained on LGE CMR using an additional variational loss.

We have provided the script of testing U-Net, PU-Net, Baseline, and BayeSeg in `demo.sh`. Please start testing these models as follows.

The setting of test directory is defined in `inference.py` as follows,
```python
if dataset == 'MSCMR' or dataset == 'ACDC':
    test_folder = "../Datasets/{}/test/{}/images/".format(dataset, sequence)
    label_folder = "../Datasets/{}/test/{}/labels/".format(dataset, sequence)
else:
    raise ValueError('Invalid dataset: {}'.format(dataset))
```
For ACDC, one need to download this dataset from its homepage, and then prepare test data as above.

To test the performance of U-Net, PU-Net, Baseline, and BayeSeg on the LGE CMR of MS-CMRSeg, please uncomment the corresponding line in `demo.sh`, and then run `sh demo.sh`.
```bash
# test Unet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model Unet --eval --dataset MSCMR --sequence LGR --resume logs/Unet/checkpoint.pth --output_dir results --device cuda

# test PUnet
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model PUnet --eval --dataset MSCMR --sequence LGR --resume logs/PUnet/checkpoint.pth --output_dir results --device cuda

# test baseline
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/baseline/checkpoint.pth --output_dir results --device cuda

# test BayeSeg
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model BayeSeg --eval --dataset MSCMR --sequence LGR --resume logs/BayeSeg/checkpoint.pth --output_dir results --device cuda

```
Here, `--sequence` can be set to C0, LGR, or T2 for MS-CMRSeg, and C0 for ACDC. For example, to test the cross-sequence segmentation performance of U-Net, PU-Net, Baseline, and BayeSeg on the T2-weighted CMR of MS-CMRSeg, please set `--sequence LGR` to `--sequence T2`.

## How to train
All models were trained using LGE CMR of MS-CMRSeg, and the root of training data is defined in `data/mscmr.py` as follows,
```python
root = Path('your/dataset/directory' + args.dataset + '/' + args.sequence)
```
Please replace `your/dataset/directory` with your own directory.

To train U-Net, PU-Net, Baseline, and BayeSeg, please uncomment the corresponding line in `demo.sh`, and run `sh demo.sh`.
```bash
# train Unet
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model Unet --batch_size 8 --output_dir logs/Unet --device cuda >train.log 2>&1 &

# train PUnet
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model PUnet --batch_size 8 --output_dir logs/PUnet --device cuda >train.log 2>&1 &

# train Baseline
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model BayeSeg --batch_size 8 --output_dir logs/Baseline --device cuda >train.log 2>&1 &

# train BayeSeg
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --model BayeSeg --batch_size 8 --output_dir logs/BayeSeg --device cuda >train.log 2>&1 &
```
Note that, to train the Baseline model, one need to comment `loss_Bayes` in `models/BayeSeg.py` as follows,
```python
weight_dict = {
        'loss_CrossEntropy': args.CrossEntropy_loss_coef,
        # 'loss_AvgDice': args.AvgDice_loss_coef,  
        # 'loss_Bayes':args.Bayes_loss_coef,
    }
```

## Citation
If our work is helpful in your research, please cite this as follows.

[1] S. Gao, H. Zhou, Y. Gao, and X. Zhuang, "Joint Modeling of Image and Label Statistics for Enhancing Model Generalizability of Medical Image Segmentation," arXiv e-print, arXiv:2206.04336, 2022. [[arXiv]](https://arxiv.org/abs/2206.04336) [[MICCAI]]()
```
@Article{Gao/BayeSeg/2022,
	title =	 {Joint Modeling of Image and Label Statistics for Enhancing Model Generalizability of Medical Image Segmentation},
	author = {Gao, Shangqi and Zhou, Hangqi and Gao, Yibo and Zhuang, Xiahai},
    journal = {	arXiv e-print, arXiv:2206.04336},
    year = 2022
}
```

Don't hesitate to contact us via [shqgao@163.com]() or [zxh@fudan.edu.cn](), if you have any questions.
