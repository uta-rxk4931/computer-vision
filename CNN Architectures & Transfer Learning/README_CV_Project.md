# Computer Vision Project – CNN Architectures & Transfer Learning

This project was developed for the CSE 4310 Computer Vision course at the University of Texas at Arlington. It explores various convolutional neural network (CNN) architectures and training techniques using the Imagenette dataset.

## Project Goals

- Build and train a basic CNN from scratch
- Implement an All Convolutional Network (All-CNN)
- Add regularization techniques like dropout and batch normalization
- Apply transfer learning using a pretrained CNN model

## Folder Structure

```
rxk4931_A03_cse4310/
├── Basic_CNN.ipynb             # Basic CNN with max-pooling
├── ALLConv.ipynb               # All-CNN using only convolutional layers
├── Regularization.ipynb        # All-CNN with dropout and weight decay
├── TransferLearning.ipynb      # Fine-tuning All-CNN with pretrained weights
├── Model Weights/              # Contains .pth and .ckpt files
├── Report.pdf                  # Summary and analysis of results
```

## How to Run

1. Make sure you have Python 3.8 or higher installed.

2. Install the required libraries:

```bash
pip install torch torchvision pytorch-lightning torchmetrics
```

3. Run the Jupyter notebooks in order from top to bottom.

4. The dataset (Imagenette) is automatically downloaded using:

```python
from torchvision.datasets import Imagenette
```

5. For `TransferLearning.ipynb`, make sure the file `allCNN_imagenette.ckpt` is placed in the same directory as the notebook.

## Output

Each notebook trains a model on the Imagenette dataset and saves the weights. Accuracy and loss metrics are logged using TensorBoard, and results are discussed in the included report.

## Author

Rency Ajit Kansagra 

University of Texas at Arlington  
