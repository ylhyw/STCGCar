
# STCGCar: Spatial Transcriptomics Clustering with Graph Contrastive Learning and Augmentation Strategy

STCGCar is a state-of-the-art framework designed for clustering spatial transcriptomics (ST) data. By leveraging graph contrastive learning, reversible networks for data augmentation, and redundancy reduction strategies, STCGCar effectively overcomes challenges in high-dimensional data clustering and achieves superior performance.

---

## Framework Overview

![STCGCar Framework](image/STCGCar.png)

The STCGCar framework consists of the following components:
1. **Graph Contrastive Learning**: Extracts robust embeddings from ST data while ensuring consistency across augmented views.
2. **Reversible Neural Networks for Data Augmentation**: Generates diverse, high-quality augmented views to improve model generalizability.
3. **Redundancy Reduction Strategy**: Mitigates feature redundancy in high-dimensional latent spaces to enhance clustering precision.

---

## System Requirements

To ensure smooth execution, the following dependencies are required:

```bash
python == 3.7
torch>=1.12.0
numpy>=1.23.3
scikit-learn>=1.0.2
pandas>=1.4.2
scanpy>=1.8.0
scipy>=1.8.0
matplotlib >= 3.5.1
```

**Hardware**:
- **Recommended GPU**: NVIDIA GTX 2080Ti or above.

---

## Parameter Settings

Below is a detailed description of the key parameters used in the STCGCar framework:

### **1. gnnlayers**
- **Description**: Number of graph convolutional layers embedding the adjacency matrix with the identity matrix.
- **Recommended Value**: 3 to 5 layers.
- **Details**: A balance between computational complexity and model expressiveness. Too many layers may cause overfitting, while too few may fail to capture the complexity of the data.

### **2. rad_cutoff**
- **Description**: Determines the neighbor network range and the number of neighbors per cell.
- **Recommended Value**: Between 6 and 15 neighbors.
- **Details**: For 10x datasets, each cell typically contains 6 neighbors. A rad_cutoff of 150 or 300 is often suitable for 10x datasets to ensure valid neighbor relationships. For other datasets, adjust rad_cutoff dynamically based on data density and structure.

### **3. alpha2**
- **Description**: Balances the redundancy loss in the model.
- **Recommended Value**: 0.5 for most new datasets.
- **Details**: Ensures stability and consistency during training.

### **4. lr (Learning Rate)**
- **Description**: Dynamically adjusted learning rate based on the loss function.
- **Default Value**: `1e-3` for all datasets during the initial phase.
- **Details**:
  - For **unlabeled datasets**: Use `1e-3`.
  - For **labeled datasets**: Use a smaller learning rate, e.g., `1e-5`, for better results.
  - Learning rate decay mechanism: Reduce the learning rate to `0.75x` every 100 epochs for better model refinement in later training stages.
  - Final range: Adjust dynamically between `1e-3` and `5e-6` based on the loss function.

---

## How to Use

### **1. Clone the Repository**
```bash
git clone https://github.com/plhhnu/STCGCar.git
cd STCGCar
```

### **2. Install Dependencies**
Install all required Python libraries:
```bash
pip install -r requirements.txt
```

### **3. Prepare Data**
- Format spatial transcriptomics data as an AnnData object.
- Ensure the dataset includes spatial coordinates and gene expression matrices.


### **4. Evaluate and Visualize Results**
- **Cluster Assignments**: Predicted spatial domains for each spot.
- **Performance Metrics**: ARI,NMI, and other clustering metrics.

---

## File Structure

```plaintext
STCGCar/
├── main.py                # Main script for training and evaluation
├── models/                # Model architectures
├── data/                  # Example datasets
├── utils/                 # Utility functions
├── outputs/               # Directory for saving results
└── README.md              # Documentation
```

---

## Results and Visualizations

STCGCar has been tested on multiple public spatial transcriptomics datasets, achieving state-of-the-art performance. Below is an example visualization of spatial clustering:

![Results](result/clusting.png)

---

## Citation

If you use STCGCar in your research, please cite:

```bibtex

```

---
