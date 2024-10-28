<div align="center">
<h1> Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation (CVPR 2024) </h1>
</div>


![image](framework.png)
Consistency learning is a central strategy to tackle unlabeled data in semi-supervised medical image segmentation (SSMIS), which enforces the model to produce consistent
predictions under the perturbation. However, most current approaches solely focus on utilizing a specific single perturbation, which can only cope with limited cases, while
employing multiple perturbations simultaneously is hard to guarantee the quality of consistency learning. In this paper, we propose an Adaptive Bidirectional Displacement (ABD) approach to solve the above challenge.

## 1. Installation

```
pip install -r requirements.txt
```
## 2. Dataset
Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) and [promise12](https://promise12.grand-challenge.org/Download/).

MS-CMRSEG19: Download from [official link](https://zmiclab.github.io/zxh/0/mscmrseg19/) or directly use preprocessed data at [onedrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155195604_link_cuhk_edu_hk/Eh0O786sCE1KuaASgpxYmj0ByM-Vqwlz3MqPdbD62Fg3KA?e=U7CltC)..
```
├── ./data
    ├── [ACDC]
        ├── [data]
        ├── test.list
        ├── train_slices.list
        ├── train.list
        └── val.list
    ├── [promise12]
        ├── CaseXX_segmentation.mhd
        ├── CaseXX_segmentation.raw
        ├── CaseXX.mhd
        ├── CaseXX.raw
        ├── test.list
        └── val.list
	├── mscmrseg19_split1/
	    ├── data/
	    │   ├── patient1_LGE.h5
	    │   ├── ...
	    │   └── slices/
	    │       ├── patient1_LGE_slice_0.h5
	    │       └── ...
	    ├── test.list
	    ├── train_slices.list
	    └── val.list
	├── mscmrseg19_split2/
	    ├── data/
	    │   ├── patient1_LGE.h5
	    │   ├── ...
	    │   └── slices/
	    │       ├── patient1_LGE_slice_0.h5
	    │       └── ...
	    ├── test.list
	    ├── train_slices.list
	    └── val.list
```

## 3. Usage
**To train a model**
```
When the paper is accepted, we will publish the training code.
``` 
**To test a model**

We have given the model of the corresponding results of our paper
```
code/model
```

if you want to test, you can follow the code below.

**Note**: For MS-CMRSEG19, the dataset is split into training and validation only. We report the averaged results on the validation sets of the two random splits.This is to follow the article [DiffRect](https://github.com/CUHK-AIM-Group/DiffRect/).

```
python test_ACDC_beta_FFT.py  # for ACDC testing
python test_promise12_beta_FFT.py  # for PROMISE12 testing
python test_MSCMR_split1.py  # for MS-CMRSEG19_split1
python test_MSCMR_split2.py  # for MS-CMRSEG19_split2


```

## Acknowledgements
Our code is largely based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [BCP](https://github.com/DeepMed-Lab-ECNU/BCP), [DiffRect](https://github.com/CUHK-AIM-Group/DiffRect/),and [ABD](https://github.com/chy-upc/ABD). Thanks for these authors for their valuable work, hope our work can also contribute to related research.
