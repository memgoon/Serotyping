# E. coli O-Serotyping with Protein LLM Embeddings

This repository provides a machine learning pipeline to classify **Escherichia coli O-serotypes** by leveraging high-dimensional sequence representations (embeddings) from the **ESM-2 (Evolutionary Scale Modeling)** protein language model.

## 🚀 Overview
Traditional serotyping methods are often labor-intensive and costly. This project explores a computational approach that uses the functional and structural information captured by **ESM-2** to achieve rapid and accurate serotype prediction. By converting protein sequences into numerical embeddings, we compare multiple supervised learning algorithms to identify the most robust model for serotype classification.

## 🛠️ Key Features
- **Embedding Extraction**: Integration of embeddings generated from the `esm2_t33_650M_UR50D` model.
- **Data Pipeline**: Automated loading of `.pt` (PyTorch) embedding files and mapping them to corresponding serotype labels.
- **Model Benchmarking**: Comparative analysis of multiple ML classifiers:
  - Random Forest
  - XGBoost
  - Extra Trees Classifier
  - Support Vector Machine (SVM)
- **Performance Evaluation**: Comprehensive metrics including Accuracy, F1-score, and Confusion Matrix visualization.

## 📁 Repository Structure
- `Serotyping_ML.ipynb`: The primary Jupyter Notebook containing the end-to-end workflow (Data preprocessing -> Training -> Evaluation).

## 📊 Performance Summary
- Ensemble methods, particularly **Extra Trees** and **Random Forest**, demonstrated high precision and recall across various O-serotypes.
- The results suggest that ESM-2 embeddings effectively capture the biological signatures required to distinguish between complex E. coli serotypes.

## ⚙️ Requirements
To run the notebook, you need the following dependencies:
- Python 3.8+
- PyTorch
- Scikit-learn
- XGBoost
- Pandas & Numpy
- Matplotlib & Seaborn

## 📝 Citation & Reference
This work is part of the research: *"Development of E. coli O-serotyping classification model based on Protein LLM"* (Under revision for **Scientific Reports**).

## 👤 Author
- **Heesu Jeong** (Seoul National University)
- Email: hs0517@snu.ac.kr
