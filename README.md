## HSIMAE：A Unified Masked Autoencoder with large-scale pretraining for Hyperspectral Image Classification

![HSIMAE-2-3](https://github.com/Ryan21wy/HSIMAE/assets/81405754/d8a1177e-a587-40d8-bebb-b63113ae1122)

![HSIMAE-8-2](https://github.com/Ryan21wy/HSIMAE/assets/81405754/6d02fbf4-d15a-4887-9c6b-e717a5c0fc57)

## ✨ Highlights
### Masked HSI Modeling with Large-Scale Pretraining
The HSIMAE was pretrained by a large-scale HSI dataset, named HyspecNet-11k, then directly finetuned on four target classification datasets.

### Multi-Scale PCA for Features Extract
To address these distributional shifts caused by the different spectral resolutions and spectral ranges between hyperspectral sensors, a MS-PCA was used to extract the multi-scale features of HSI spectra and transform the raw spectra into fixed-length features.

### Dual-branch finetuning to leverage unlabeled data of target dataset
Dual-branch finetuning framework was proposed by using an extra unlabeled branch to further adapted the model to the distributions of the target dataset and suppressed the overfitting issue.

## 🔨 Installation
  
1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)   
2. Install [Git](https://git-scm.com/downloads)  
4. Open commond line, create environment and enter with the following commands:  

        conda create -n HSIMAE python=3.8
        conda activate HSIMAE

5. Clone the repository and enter:  

        git clone https://github.com/Ryan21wy/HSIMAE.git
        cd HSIMAE

6. Install dependency with the following commands:
        
        pip install -r requirements.txt

## 🚀 Checkpoint

The pretrained models of HSIMAE are provided in [Hugging Face](https://huggingface.co/RyanWy/HSIMAE).

## 🧐 Evaluation Results

### Pretraining Dataset:
HySpecNet-11k: [HySpecNet-11k - A Large-Scale Hyperspectral Benchmark Dataset (rsim.berlin)](https://hyspecnet.rsim.berlin/)

### Classification Dataset:
Salinas: [Salinas scene](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene)

Pavia University: [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)

Houston 2013: [2013 IEEE GRSS Data Fusion Contest](https://hyperspectral.ee.uh.edu/?page_id=459)

WHU-Hi-LongKou: [WHU-Hi: UAV-borne hyperspectral and high spatial resolution (H2) benchmark datasets](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)

### Results

Overall accuracy of four HSI classification datasets. The training set and validation set contained **50 random samples per class** , respectively, and the remaining samples were considered as the test set.

|Model|Size|Salinas|Pavia University|Houston 2013|WHU-Hi-LongKou|Average|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|HSIMAE-B|12.5M|94.20|**91.90**|92.21|98.15|94.12|
|HSIMAE-L|38.5M|94.39|91.28|92.66|**98.22**|94.14|
|HSIMAE-H|152M|**94.41**|91.43|**92.99**|98.02|**94.21**|

## 🧑‍💻 Contact

Wang Yue   
E-mail: ryanwy@csu.edu.cn 
