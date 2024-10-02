## ❄️ PerSense Overview
Official Implementation of the paper **"PerSense: Personalized Instance Segmentation in Dense Images"** 🚩

### Abstract
Leveraging large-scale pre-training, vision foundational models showcase notable performance benefits. Recent segmentation algorithms for natural scenes have advanced significantly. However, existing models still struggle to automatically segment personalized instances in dense and crowded scenarios, where severe occlusions, scale variations, and background clutter pose a challenge to accurately delineate densely packed instances of the target object. To address this, we propose **PerSense**, an end-to-end, training-free, and model-agnostic one-shot framework for **Per**sonalized instance **S**egmentation in d**ense** images. Towards developing this framework, we make the following core contributions. (a) We develop a new baseline capable of automatically generating instance-level point prompts via proposing a novel Instance Detection Module (IDM) that leverages density maps, encapsulating spatial distribution of objects in an image. (b) To mitigate false positives within generated point prompts, we design Point Prompt Selection Module (PPSM). Both IDM and PPSM transform density maps into personalized precise point prompts for instance-level segmentation and offer a seamless integration in our model-agnostic framework. (c) We introduce a feedback mechanism which enables PerSense to improve the accuracy of density maps by automating the exemplar selection process for density map generation.(d) To promote algorithmic advances and effective tools for this relatively underexplored task, we introduce PerSense-D, a diverse dataset exclusive to personalized instance segmentation in dense images. Our extensive experiments establish PerSense superiority in dense scenarios by achieving an mIoU of **71.61%** on PerSense-D, outperforming recent SOTA models by significant margins of **+47.16%**, **+42.27%**, **+8.83%**, and **+5.69%**. Additionally, our qualitative findings demonstrate the adaptability of our framework to images captured in-the-wild.

![intro_fig_arxiv](https://github.com/user-attachments/assets/ece19aeb-5be6-462e-b011-29d33ddc6951)

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

![Updated_Main Fig](https://github.com/user-attachments/assets/025c2927-971b-47cf-a957-46579694dd55)


## 🛠️ Requirements

### Installation
Similar to SAM and PerSAM, our code requires pytorch>=1.7 and torchvision>=0.8. For compatibility check [here](https://pytorch.org/get-started/locally/).
Clone the repo and create conda environment following the instructions given below:

    git clone https://github.com/Muhammad-Ibraheem-Siddiqui/PerSense.git
    cd PerSense

    conda create -n persense python=3.8
    conda activate persense
    
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    (You can also change it as per your platform following the link given above)

    pip install -r requirements.txt

### Dataset
Please download our **PerSense-D dataset** from [here](https://drive.google.com/file/d/1ku_tY3VflD-K9-xeSocgQjcd2C6oj29g/view?usp=sharing). Unzip the dataset and organize it as follows:

    data/
    |-- Images/

### 🔩 Model Weights
Please download pretrained weights of SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
Also, download weights for DSALVANet pretrained on FSC-147 from [here](https://drive.google.com/file/d/1julzH9MJSK1xTGchb1r0CXdZ2wzF5-Kp/view?usp=sharing) and weights for GroungdingDINO from [here](https://drive.google.com/file/d/13rV6dzRwWCVZYWpnmiaVwRDIDC28d82g/view?usp=sharing). Download weights for CounTR pretrained on FSC-147 dataset from [here](https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view?pli=1). For ViPLLaVA (VLM) weights are automatically fetched through the code.

    data/
    sam_vit_h_4b8939.pth

    DSALVANet/checkpoints/
    checkpoint_200.pth

    GroundingDINO/weights/
    groundingdino_swint_ogc.pth

    CounTR/output_allnew_dir/
    FSC147.pth

## 🏃‍♂️ Getting Started

#### PerSense
To evaluate PerSense on DSALVANet, just run the following command: 

    python persense.py (add argument '--visualize True' for visualizing the mask overlaid on original image)

To evaluate PerSense on CounTR, just run the following command: 

    python persense.py --fsoc countr (add argument --visualize True for visualizing masks overlaid on original image)

#### Grounded-SAM
To evaluate Grounded-SAM in our scenario, just run the following command:

    python groundedsam.py (add argument '--visualize True' for visualizing the mask overlaid on original image)

#### Matcher
Follow installation instructions given [here](https://github.com/aim-uofa/Matcher). Organize PerSense-D as *Matcher/data/Images/*. Place *matcher_test.py* and *eval_miou.py* under Matcher/. Run the following command to test Matcher on PerSense-D dataset:

    python matcher_test.py

#### PerSAM
persense conda environment can be used for PerSAM. Just clone the PerSAM repo available [here](https://github.com/ZrrSkywalker/Personalize-SAM). Place *PerSense-D* in the default dataset directory and run following:

    persam_f_multi_obj.py

#### Evaluate mIoU
To evaluate mIoU, just run the following command:

    python eval_miou.py --pred_path PerSense or groundedsam or Matcher or persam

## 👀 How the Output Looks?

![Qualitative results_arxiv](https://github.com/user-attachments/assets/371689b0-c9b8-4fc1-af3a-ba9ea6c81f1c)



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



