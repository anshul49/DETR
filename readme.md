# VinDR-CXR Chest Disease Detection using DETR Model

This project aims to detect various chest diseases from chest X-ray images using the DETR (Detection Transformer) model on the VinDR-CXR dataset. The project includes training the DETR model, evaluating its performance, and plotting FROC (Free-response Receiver Operating Characteristic) curves to assess the model's sensitivity and false positive rate.

## Dataset

The dataset used is the VinDR-CXR dataset, which includes annotated chest X-ray images with the following findings:
- Calcification
- Cardiomegaly
- ILD (Interstitial Lung Disease)
- Pneumothorax
- Other lesions
- Pleural effusion
- Pulmonary fibrosis
- Infiltration
- Pleural thickening

## Installation

To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/vindr-detr.git
cd vindr-detr
pip install -r requirements.txt
