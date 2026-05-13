# Fruit Disease Detection System
Naan-Nee Architecture — Hybrid ResNet50 + Vision Transformer

## Developers
- V. Rohit
- P. Kavya

## College
Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology
B.Tech Computer Science and Engineering — Batch 2022-2026
Final Year Major Project

## Project Overview
This project detects fruit and plant diseases from leaf images
using a Hybrid Deep Learning model combining ResNet50 and
Vision Transformer (ViT). The model achieves 99.74% accuracy
on the PlantVillage dataset — surpassing the IEEE paper
benchmark of 98.37%.

## Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 99.74% |
| Precision | 1.00   |
| Recall    | 1.00   |
| F1 Score  | 1.00   |


## Dataset
- Name     : PlantVillage Dataset
- Images   : 54,305
- Classes  : 38 disease categories
- Source   : https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

## Project Structure
fruit_disease_detection/
├── config.py        - All settings
├── dataset.py       - Data loading and augmentation
├── model.py         - Hybrid ResNet50 + ViT architecture
├── train.py         - Training loop
├── evaluate.py      - Evaluation and metrics
├── predict.py       - Single image prediction
├── utils.py         - Helper functions
├── app.py           - Web interface (Gradio)
├── data/            - Dataset folder
├── checkpoints/     - Saved model weights
└── results/         - Plots and logs

## How to Run

### Install dependencies
pip3 install -r requirements.txt

### Train the model
python3 train.py

### Evaluate the model
python3 evaluate.py

### Predict on a single image
python3 predict.py path/to/image.jpg

### Launch web interface
python3 app.py
Then open http://127.0.0.1:7860

## Technologies Used
- PyTorch
- ResNet50 pretrained on ImageNet
- Vision Transformer vit_small_patch16_224
- timm library
- Gradio web interface
- Mac M2 with MPS acceleration
