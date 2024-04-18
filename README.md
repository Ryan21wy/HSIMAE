## HSIMAE：A Unified Masked Autoencoder with large-scale pretraining for Hyperspectral Image Classification

![figure 2](https://github.com/Ryan21wy/HSIMAE/assets/81405754/d1a7e493-0390-45c2-8cdb-0c25fe771451)

![figure 4](https://github.com/Ryan21wy/HSIMAE/assets/81405754/489bb410-0ada-44af-ab77-4d0626ed9669)

## ✨ Highlights
### Large-Scale and Diverse Dataset for HSI Pretraining
A large and diverse HSI dataset named HSIHybrid was curated for large-scale HSI pre-training. It consisted of 15 HSI datasets from different hyperspectral sensors. After splitting into image patches, a total of **4 million** HSI patches with a spatial size of 9×9 were obtained.

### New MAE Architecture for HSI domain
A modified MAE named HSIMAE that utilized separate spatial-spectral encoders followed by fusion blocks to learn spatial correlation and spectral correlation of HSI data was proposed.

### Dual-branch finetuning to leverage unlabeled data of target dataset
A dual-branch fine-tuning framework was introduced to leverage the unlabeled data of the downstream HSI dataset and suppressed overfitting on small training samples.

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

The pre-training dataset and pretrained models of HSIMAE are provided in [Hugging Face](https://huggingface.co/RyanWy/HSIMAE).

Because it is too big, HySpecNet-11k need be downloaded from [HySpecNet-11k - A Large-Scale Hyperspectral Benchmark Dataset (rsim.berlin)](https://hyspecnet.rsim.berlin/)

## 🧐 Evaluation Results

### Classification Dataset:
Salinas: [Salinas scene](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene)

Pavia University: [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)

Houston 2013: [2013 IEEE GRSS Data Fusion Contest](https://hyperspectral.ee.uh.edu/?page_id=459)

WHU-Hi-LongKou: [WHU-Hi: UAV-borne hyperspectral and high spatial resolution (H2) benchmark datasets](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)

### Results

Overall accuracy of four HSI classification datasets. The training set and validation set contained **5/10/15/20 random samples per class** , respectively, and the remaining samples were considered as the test set.

|Training Samples|Salinas|Pavia University|Houston 2013|WHU-Hi-LongKou|Average|
|:---:|:---:|:---:|:---:|:---:|:---:|
|5|92.99|87.00|83.89|96.16|90.01|
|10|95.14|96.02|90.14|97.64|94.74|
|15|96.51|97.09|94.52|98.08|96.55|
|20|96.62|97.44|95.65|98.41|97.03|

## 🧑‍💻 Contact

Wang Yue   
E-mail: ryanwy@csu.edu.cn 
