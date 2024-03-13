# Vision-Transformers-for-Small-Histological-Datasets-Learned-Through-Knowledge-Distillation
Source code for paper "Vision Transformers for Small Histological Datasets Learned Through Knowledge Distillation", published at 27th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2023)

Preprint =  https://arxiv.org/pdf/2305.17370.pdf 
published version = https://link.springer.com/chapter/10.1007/978-3-031-33380-4_13

# Requirements 
- Pytorch
- Timm
- vit_pytorch
- Pandas
- Matplotlib
- Scikit-learn
- Yagmail

# Abstract
Computational Pathology (CPATH) systems have the potential to automate diagnostic tasks. However, the artifacts on the digitized histological glass slides, known as Whole Slide Images (WSIs), may hamper the overall performance of CPATH systems. Deep Learning (DL) models such as Vision Transformers (ViTs) may detect and exclude artifacts before running the diagnostic algorithm. A simple way to develop robust and generalized ViTs is to train them on massive datasets. Unfortunately, acquiring large medical datasets is expensive and inconvenient, prompting the need for a generalized artifact detection method for WSIs. In this paper, we present a student-teacher recipe to improve the classification performance of ViT for the air bubbles detection task. ViT, trained under the student-teacher framework, boosts its performance by distilling existing knowledge from the high-capacity teacher model. Our best-performing ViT yields 0.961 and 0.911 F1-score and MCC, respectively, observing a 7% gain in MCC against stand-alone training. The proposed method presents a new perspective of leveraging knowledge distillation over transfer learning to encourage the use of customized transformers for efficient preprocessing pipelines in the CPATH systems.

<img width="603" alt="image" src="https://github.com/NeelKanwal/Vision-Transformers-for-Small-Histological-Datasets-Learned-Through-Knowledge-Distillation/assets/52494244/a04e03e9-11fd-4bd6-9ecd-ffb62b7d2c0a">

# Result

<img width="592" alt="image" src="https://github.com/NeelKanwal/Vision-Transformers-for-Small-Histological-Datasets-Learned-Through-Knowledge-Distillation/assets/52494244/0cfd1ca2-21aa-4513-a157-1b84d8f36674">

<img width="1153" alt="image" src="https://github.com/NeelKanwal/Vision-Transformers-for-Small-Histological-Datasets-Learned-Through-Knowledge-Distillation/assets/52494244/818bb8ad-3f3e-42a7-8cdf-da78aaf5d9d0">

# Dataset
The dataset is available at https://zenodo.org/records/10809442. 

If you plan to use your dataset, then organize it with the following structure: An example of a blood artifact is shown below.

Use D40x and the folders with related artifacts and artifact_free images in the following order.

```
- path_to\airbubble_dataset
      - training
           -- artifact_free
            -- bubble
      - validation
            -- artifact_free
            -- bubble
       - test
            -- artifact_free
            -- bubble
```

# How to use the code
- Use train_dcnns.py, train_transformers and distillation.py to train models as reported in Tables 1 and 2. Hyperparametric selection can be defined inside the header of the file.

If you use this code, then please cite our paper.
```
@inproceedings{kanwal2023vision,
  title={Vision Transformers for Small Histological Datasets Learned Through Knowledge Distillation},
  author={Kanwal, Neel and Eftest{\o}l, Trygve and Khoraminia, Farbod and Zuiverloon, Tahlita CM and Engan, Kjersti},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={167--179},
  year={2023},
  organization={Springer}
}
```
