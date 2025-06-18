<div align="center">
<h1> &beta;-FFT: Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation(CVPR 2025) </h1>
</div>

 Co-training has achieved significant success in the field of semi-supervised learning(SSL); however, the homogenization phenomenon, which arises from multiple models tending towards similar decision boundaries, remains inadequately addressed. To tackle this issue, we propose a novel algorithm called $\beta$-FFT from the perspectives of data processing and training structure. In data processing, we apply diverse augmentations to input data and feed them into two sub-networks. To balance the training instability caused by different augmentations during consistency learning, we introduce a nonlinear interpolation technique based on the Fast Fourier Transform (FFT). By swapping low-frequency components between variously augmented images, this method not only generates smooth and diverse training samples that bridge different augmentations but also enhances the model's generalization capability while maintaining consistency learning stability.
In training structure, we devise a differentiated training strategy to mitigate homogenization in co-training. Specifically, we use labeled data for additional training of one model within the co-training framework, while for unlabeled data, we employ linear interpolation based on the **Beta (&beta;)** distribution as a regularization technique in additional training. This approach allows for more efficient utilization of limited labeled data and simultaneously improves the model's performance on unlabeled data, optimizing overall system performance.

![image](framework.png)


## Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation(CVPR 2025)

## 1. Installation

```
conda create -n beta python=3.8
conda activate beta
pip install -r requirements.txt
```
## 2. Dataset
Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) and [promise12](https://promise12.grand-challenge.org/Download/).

MS-CMRSEG19: Download from [official link](https://zmiclab.github.io/zxh/0/mscmrseg19/) or directly use preprocessed data at [onedrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155195604_link_cuhk_edu_hk/Eh0O786sCE1KuaASgpxYmj0ByM-Vqwlz3MqPdbD62Fg3KA?e=U7CltC)..
```
├── ./train_MSCMRSEG
├── ./train_promise12
├── ./train_ACDC
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
    ├── [MS_CMRSEG]
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
├──ACDC_train_label_7:
    cd train_ACDC
    python beta_fft_label7.py

├──ACDC_train_label_3:
    cd train_ACDC
    python beta_fft_label3.py

├──MSCMRSEG_spilt1:
   cd train_MSCMRSEG
   python train_beta_fft1.py

├──MSCMRSEG_spilt2:
   cd train_MSCMRSEG
   python train_beta_fft2.py

├──promise12:
  cd train_promise12
  python train_promise12_bcp_fft.py


``` 
**To test a model**

We have given the model of the corresponding results of our paper
```
best_results/
```

if you want to test, you can follow the code below.

**Note**: For MS-CMRSEG19, the dataset is split into training and validation only. We report the averaged results on the validation sets of the two random splits.This is to follow the article [DiffRect](https://github.com/CUHK-AIM-Group/DiffRect/).

```
python test_ACDC_beta_FFT.py  # for ACDC testing
python test_PROMISE12.py  # for PROMISE12 testing

*******************Replace the paths of split1 and split2 respectively.
python test_2D.py  # for MS-CMRSEG19_split1
python test_2D.py  # for MS-CMRSEG19_split2

```

## Acknowledgements
Our code is largely based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [BCP](https://github.com/DeepMed-Lab-ECNU/BCP), [DiffRect](https://github.com/CUHK-AIM-Group/DiffRect/),[MixUp](https://github.com/facebookresearch/mixup-cifar10),and [ABD](https://github.com/chy-upc/ABD). Thanks for these authors for their valuable work, hope our work can also contribute to related research.


**Email:** huming708@gmail.com
