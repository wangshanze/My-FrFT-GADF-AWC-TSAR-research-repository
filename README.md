# 📌 My-FrFT-GADF-AWC-TSAR-Research-Repository
## 😀 Significance and Purpose of This Repository

This repository is used to store the source code of my **FrFT-GADF-AWC-TSAR model**, which is designed for **fault diagnosis of aero-engine bearings**.

The work has recently undergone a round of revisions and is currently **under consideration for publication in the journal *Measurement***. It has not yet been accepted at this stage.

I have **voluntarily made the source code publicly available**.  
Any comments, suggestions, or constructive feedback are **warmly welcome**. 🙌


## 🎯 Repository Structure

```text
.
├── data/                  # Raw data
│   ├── hit/               # HIT dataset
│   └── hust/              # HUST dataset
├── draw/                  # Visualization and plotting scripts
├── model/                 # Model definitions
├── research-code/         # Research code
│   └── tsne_results/      # t-SNE dimensionality reduction results
├── Result/                # Main experimental results
│   ├── hit/
│   └── hust/
├── Result_AB/             # Best .pth models
└── utils/                 # Common utility functions
````
---

## ❓ Requirements
matplotlib==3.7.5
numpy==1.24.3
numpy==1.24.4
pandas==2.0.3
pyts==0.13.0
PyWavelets==1.4.1
scikit_learn==1.3.0
scikit_learn==1.3.2
scipy==1.10.1
torch==2.3.1+cu121
torchvision==0.18.1+cu121

---

## ❓ Data Description and Notes

The datasets used in this work are provided by **Harbin Institute of Technology (HIT)** and **Huazhong University of Science and Technology (HUST)**.
The corresponding DOIs are listed below:

* **HIT Dataset**
  DOI: `10.37965/jdmd.2023.314`

* **HUST Dataset**
  DOI: `10.1016/j.ress.2024.109964`

Due to the large size of the datasets, it is **not convenient to upload them directly to GitHub**.
Therefore, the processed data in the `data/` directory has been uploaded to cloud storage.

You can **download the data and replace the local `data/` directory directly**.
The data has already been preprocessed and is **ready for immediate use**.

### 📦 Google Drive

[https://drive.google.com/drive/folders/1QyeHKVDv-vLnRQr4rin1gfft9_qagqV6?usp=drive_link](https://drive.google.com/drive/folders/1QyeHKVDv-vLnRQr4rin1gfft9_qagqV6?usp=drive_link)

### 📦 Xunlei Drive

Link: [https://pan.xunlei.com/s/VOgedWwFzdrA2TzzB85NHINmA1](https://pan.xunlei.com/s/VOgedWwFzdrA2TzzB85NHINmA1)
Extraction Code: `rpce`

---

## 🤔 How to Use?

Very simple! 🚀

1. Download the dataset from the cloud storage.
2. Place the data into the `data/` directory.
3. Navigate to the `research-code/` folder.

Inside `research-code/`, you will find:

* **Jupyter notebooks** for interactive experiments
* **`main.py`**, which is used for **performance comparison across different models**

You may run **either option** depending on your needs.

---

## ✨ Acknowledgement

I would like to express my sincere gratitude to the **editors and reviewers of the journal *Measurement*** for their hard work, valuable comments, and insightful suggestions. 

Special thanks are also extended to **my supervisor** for continuous guidance and support.

---

## 📑 Future Work

In the future, this work will be extended to develop **more intelligent multi-order fractional Fourier transform fusion algorithms**.
I believe such studies are both **necessary and meaningful** for advancing intelligent fault diagnosis.

Thank you very much for your interest and time! 😊


