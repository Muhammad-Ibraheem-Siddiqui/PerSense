## ❄️ PerSense Overview
Official Implementation of the paper **"PerSense: Personalized Instance Segmentation in Dense Images"** 🚩

### Abstract
Leveraging large-scale pre-training, vision foundational models showcase notable performance benefits. While recent years have witnessed significant advancements in segmentation algorithms, existing models still face challenges to automatically segment personalized instances in dense and crowded scenarios. The primary factor behind this limitation stems from bounding box-based detections, which are constrained by occlusions, background clutter, and object orientation, particularly when dealing with dense images. To this end, we propose \textbf{PerSense}, an end-to-end, training-free, and model-agnostic one-shot framework to address the Personalized instance Segmentation in dense images. Towards developing this framework, we make following core contributions. (a) We propose an Instance Detection Module (IDM) and leverage a Vision-Language Model, a grounding object detector, and a few-shot object counter (FSOC) to realize a new baseline. (b) To tackle false positives within candidate point prompts, we design Point Prompt Selection Module (PPSM). Both IDM and PPSM transform density maps from FSOC into personalized instance-level point prompts for segmentation and offer a seamless integration in our model-agnostic framework. (c) We introduce a feedback mechanism which enables PerSense to harness the full potential of FSOC by automating the exemplar selection process. (d) To promote algorithmic advances and effective tools for this relatively underexplored task, we introduce PerSense-D, a dataset exclusive to personalized instance segmentation in dense images. We validate the effectiveness of PerSense on the task of personalized instance segmentation in dense images on PerSense-D and comparison with SOTA. Additionally, our qualitative findings demonstrate the adaptability of our framework to images captured in-the-wild.

  ![intro_fig_latest](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense/assets/142812051/690a2aec-e677-4805-a8d7-3333e0f5f228)

## 🔥 News
* We release the code for **PerSense** 🚀
* We release a new dataset for Personalized one-shot Segmentation in Dense Images, **PerSense-D**🚀
* PerSense paper is released [arXiv Link](https://arxiv.org/abs/2405.13518)
## 🌟 Highlight
We introduce **PerSense** 🚀 for **Personalized Instance Segmentation** in **Dense Images**. 

👑 **End-to-End**  
❄️ **Training-free**  
💡 **Model-agnostic**  
🎯 **One-shot Framework**  


![main_fig_new](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense/assets/142812051/64f1141a-e0b2-4171-ab1c-cc4b8f6b1d07)


## 🛠️ Requirements

### Installation
Similar to SAM and PerSAM, our code requires pytorch>=1.7 and torchvision>=0.8. For compatibility check [here](https://pytorch.org/get-started/locally/).
Clone the repo and create conda environment following the instructions given below:

    git clone https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense.git
    cd PerSense

    conda create -n persense python=3.8
    conda activate persense

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
    (Change it as per your platform following the link given above)

    pip install -r requirements.txt

### Dataset
Please download our **PerSense-D dataset** from [here](https://drive.google.com/file/d/1ku_tY3VflD-K9-xeSocgQjcd2C6oj29g/view?usp=sharing). Unzip the dataset and organize it as follows:

    data/
    |-- Images/

### 🔩 Model Weights
Please download pretrained weights of SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
Also, download weights for DSALVANet pretrained on FSC-147 from [here](https://drive.google.com/file/d/1julzH9MJSK1xTGchb1r0CXdZ2wzF5-Kp/view?usp=sharing) and weights for GroungdingDINO from [here](https://drive.google.com/file/d/13rV6dzRwWCVZYWpnmiaVwRDIDC28d82g/view?usp=sharing). For ViPLLaVA (VLM) weights are automatically fetched through the code.

    data/Images/
    sam_vit_h_4b8939.pth

    DSALVANet/checkpoints/
    checkpoint_200.pth

    GroundingDINO/weights/
    groundingdino_swint_ogc.pth

## 🏃‍♂️ Getting Started

#### PerSense
To evaluate PerSense, just run the following command: 

    python persense.py (add argument '--visualize True' for visualizing the mask overlaid on original image)

#### Grounded-SAM
To evaluate Grounded-SAM in our scenario, just run the following command:

    python groundedsam.py (add argument '--visualize True' for visualizing the mask overlaid on original image)

#### Evaluate mIoU
To evaluate mIoU, just run the following command:

    python eval_miou.py --pred_path PerSense or groundedsam

## 👀 How the Output Looks?

![2](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense/assets/142812051/e3bdb921-0807-4cf3-a7bf-c5360dd34a27)




## ❤️ Acknowledgement 
Our repo benefits from [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM/tree/main?tab=readme-ov-file), [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [DSALVANet](https://github.com/kadvinj/DSALVANet?tab=readme-ov-file) and [ViPLLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA/tree/main). Thanks for their great work.

## License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)  

**Usage and License Notices**: The data, model and pretrained checkpoints are intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## ✒️ Citation
    @article{siddiqui2024persense,
    title={PerSense: Personalized Instance Segmentation in Dense Images},
    author={Siddiqui, Muhammad Ibraheem and Sheikh, Muhammad Umer and Abid, Hassan and Khan, Muhammad Haris},
    journal={arXiv preprint arXiv:2405.13518},
    year={2024}
  }



