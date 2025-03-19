<div align="center">
<h1> &beta;-FFT: Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation(CVPR 2025) </h1>
</div>

Co-training has achieved significant success in the field of semi-supervised learning; however, the *homogenization phenomenon*, which arises from multiple models tending towards similar decision boundaries, remains inadequately addressed. To tackle this issue, we propose a novel algorithm called **&beta;-FFT** from the perspectives of data diversity and training structure.First, from the perspective of data diversity, we introduce a nonlinear interpolation method based on the **Fast Fourier Transform (FFT)**. This method generates more diverse training samples by swapping low-frequency components between pairs of images, thereby enhancing the model's generalization capability. Second, from the structural perspective, we propose a differentiated training strategy to alleviate the homogenization issue in co-training. In this strategy, we apply additional training with labeled data to one model in the co-training framework, while employing linear interpolation based on the **Beta (&beta;)** distribution for the unlabeled data as a regularization term for the additional training. This approach enables us to effectively utilize the limited labeled data while simultaneously improving the model's performance on unlabeled data, ultimately enhancing the overall performance of the system.Experimental results demonstrate that **&beta;-FFT** outperforms current state-of-the-art (SOTA) methods on three public medical image datasets.

![image](framework.png)




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
**I will organize my code after submitting to NeurIPS 2025, and plan to make the code publicly available by the end of June.**
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
