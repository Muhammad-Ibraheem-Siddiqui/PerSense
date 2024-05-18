## ❄️ PerSense
Official Implementation of the paper **"PerSense: Personalized Instance Segmentation in Dense Images"** 🚩

  ![intro_fig_latest](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense/assets/142812051/690a2aec-e677-4805-a8d7-3333e0f5f228)

## 🔥 News
* We release the code for **PerSense** 🚀
* We release a new dataset for Personalized one-shot Segmentation in Dense Images, **PerSense-D**🚀
## 🌟 Highlight
We introduce **PerSense** 🚀 for **Personalized Instance Segmentation** in **Dense Images**. 

👑 **End-to-End**  
❄️ **Training-free**  
💡 **Model-agnostic**  
🎯 **One-shot Framework**  




![main_fig_new](https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense/assets/142812051/6dd1a7df-2991-4570-8a9b-5ab903b6266a)

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
Please download our **PerSense-D dataset** from [here](https://drive.google.com/file/d/1rpPCPYz0lGUs_DEJ2-ZllNyHvhvibhtr/view?usp=drive_link). Unzip the dataset and organize it as follows:

    data/
    |-- Images/

### 🔩 Model Weights
Please download pretrained weights of SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
Also, download weights for DSALVANet pretrained on FSC-147 from [here](https://drive.google.com/file/d/1julzH9MJSK1xTGchb1r0CXdZ2wzF5-Kp/view?usp=drive_link) and weights for GroungdingDINO from [here](https://drive.google.com/file/d/13rV6dzRwWCVZYWpnmiaVwRDIDC28d82g/view?usp=drive_link). For ViPLLaVA (VLM) weights are automatically fetched through the code.

    data/
    |-- Images/
    sam_vit_h_4b8939.pth

    DSALVANet/checkpoints
    checkpoint_200.pth

    GroundingDINO/weights
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

## ❤️ Acknowledgement 
Our repo benefits from [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM/tree/main?tab=readme-ov-file), [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [DSALVANet](https://github.com/kadvinj/DSALVANet?tab=readme-ov-file) and [ViPLLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA/tree/main). Thanks for their great work.

## License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)  

**Usage and License Notices**: The data, model and pretrained checkpoints are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## ✒️ Citation



