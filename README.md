## **PerSense: Personalized Instance Segmentation in Dense Images** [BMVC 2025 🔥]

### Abstract
The emergence of foundational models has significantly advanced segmentation approaches. However, challenges still remain in dense scenarios, where occlusions, scale variations, and clutter impede precise instance delineation. To address this, we propose **PerSense**, an end-to-end, training-free, and model-agnostic one-shot framework for **Per**sonalized instance **S**egmentation in d**ense** images. We start with developing a new baseline capable of automatically generating instance-level point prompts via proposing a novel Instance Detection Module (IDM) that leverages density maps (DMs), encapsulating spatial distribution of objects in an image. To reduce false positives, we design the Point Prompt Selection Module (PPSM), which refines the output of IDM based on an adaptive threshold. Both IDM and PPSM seamlessly integrate into our model-agnostic framework. Furthermore, we introduce a feedback mechanism that enables PerSense to improve the accuracy of DMs by automating the exemplar selection process for DM generation. Finally, to advance research in this relatively underexplored area, we introduce PerSense-D, an evaluation benchmark for instance segmentation in dense images. Our extensive experiments establish PerSense's superiority over SOTA in dense settings.

![intro_fig_ICLRmerged1and2_BMVC2](https://github.com/user-attachments/assets/158aa5b3-ac1f-4644-8e20-622d12cbb1eb)


## 🔥 News
* Excited to announce acceptance of PerSense at BMVC 2025 🚀
* We release the code for **PerSense** 🚀
* We release a new dataset for Personalized one-shot Segmentation in Dense Images, **PerSense-D**🚀
* PerSense paper is released [arXiv Link](https://arxiv.org/abs/2405.13518)
## 🌟 Highlight
We introduce **PerSense** 🚀 for **Personalized Instance Segmentation** in **Dense Images**. 

👑 **End-to-End**  
❄️ **Training-free**  
💡 **Model-agnostic**  
🎯 **One-shot Framework**  

![Updated_Main_Fig3](https://github.com/user-attachments/assets/68fa7f49-9c5a-47a5-9eb9-004dbff01137)



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

## 👀 How it Looks?

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



