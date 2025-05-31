# CircularSegmentation

This project focuses on detecting and segmenting circular objects — specifically sports balls — within images from the COCO 2017 dataset. Using a semantic segmentation deep learning approach built on a MobileNetV2 backbone, the model predicts precise pixel-wise masks that delineate the objects.

Further, it extracts geometric parameters of these segmented objects by fitting minimum enclosing circles around the detected masks, enabling accurate localization and size estimation of the circular objects. This pipeline combines classical image processing techniques with modern deep learning to achieve robust and interpretable results on complex real-world images.

The project showcases:
```
- Data preprocessing and mask generation from polygonal COCO annotations,
- Custom dice loss optimized semantic segmentation model,
- Comprehensive training with callbacks for early stopping and learning rate adjustment,
- Evaluation using multiple metrics including pixel-wise IoU and Dice coefficient,
- Visualization of segmentation masks alongside circle fits for qualitative analysis.
```

## Features

- Custom segmentation pipeline tailored for circular object detection in complex images.
- Utilizes COCO dataset's polygonal annotations to generate precise object masks.
- Deep learning model based on MobileNetV2 backbone with dice loss for improved mask accuracy.
- Efficient preprocessing including resizing and grayscale conversion for training inputs.
- Extraction of center coordinates and radius of circular contours from predicted masks.
- Robust training workflow with early stopping, learning rate reduction, and model checkpointing.
- Quantitative evaluation using pixel-wise IoU, Dice coefficient, accuracy, and ROC-AUC.
- Visualizations combining original images, ground truth masks, predicted masks, and circle fits.

## Tech Stack

1. **Python 3.8+** :  Core programming language.
2. **TensorFlow 2.x / Keras** :  Deep learning framework for model building and training.
3. **OpenCV** : Image processing and contour analysis.
4. **pycocotools** : Handling COCO dataset annotations.
5. **NumPy & Pandas** : Data manipulation and array operations.
6. **Matplotlib & Seaborn** : Visualization of results and metrics.
7. **scikit-learn** : Metrics and evaluation utilities.

## Folder Structure
```
CircularSegmentation/
│
├── Main Colab Notebook
├── Plots
├── requirements.txt
├── README.md
└── .gitignore

```

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**
```
git clone https://github.com/yuvrajtiwary-bitmesraece/CircularSegmentation.git
cd CircularSegmentation
```

2. **Create and activate a virtual environment**
```
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install required dependencies**
```
pip install -r requirements.txt
```

## How To Run?

You can explore the complete workflow (training + testing + visualization) using the Colab notebook: 

[Run on Colab](https://colab.research.google.com/drive/1yGPW90mz5yZxgor2h7v8lTCArJ-QXEBH?usp=sharing)

## Results

This section presents the key evaluation metrics and visual insights derived from the trained model. These results demonstrate the model's effectiveness in segmenting circular objects (e.g., sports balls) from complex scenes.


### Accuracy Curve

This plot shows the training and validation accuracy across epochs. A steady rise and convergence between both lines indicate that the model has learned relevant features and is generalizing well without overfitting.

![Accuracy Curve CircSeg](https://raw.githubusercontent.com/yuvrajtiwary-bitmesraece/CircularSegmentation/main/Accuracy%20Curve%20CircSeg.png)

### Loss Curve

Displays training and validation loss values. The decreasing trend with stabilization reflects successful model optimization using Dice loss, which is suitable for handling class imbalance in segmentation tasks.

![Loss Curve CircSeg](https://raw.githubusercontent.com/yuvrajtiwary-bitmesraece/CircularSegmentation/main/Loss%20Curve%20CircSeg.png)

### Confusion Matrix (Pixel Level)

This matrix gives a pixel-level classification breakdown of true positives, false positives, true negatives, and false negatives. The strong diagonal presence highlights good prediction fidelity.

![Confusion Matrix CircSeg](https://raw.githubusercontent.com/yuvrajtiwary-bitmesraece/CircularSegmentation/main/Confusion%20Matrix%20CircSeg.png)

###  Mask and Circle Visualization

Visual comparison of the original grayscale image, ground-truth mask, predicted circle mask, and extracted geometric circle overlays. This qualitative analysis validates the model’s ability to correctly localize and parameterize circular regions.

![Mask and Circle Visualization CircSeg](https://raw.githubusercontent.com/yuvrajtiwary-bitmesraece/CircularSegmentation/main/Mask%20and%20Circle%20Visualization%20CircSeg.png)

### ROC Curve

Illustrates the model’s ability to distinguish foreground (circle) vs. background at various thresholds. The high AUC (Area Under Curve) confirms robust pixel-wise discrimination and confident predictions.

![ROC Curve CircSeg](https://raw.githubusercontent.com/yuvrajtiwary-bitmesraece/CircularSegmentation/main/ROC%20Curve%20CircSeg.png)

## Thank You

Grateful for your time and attention — it truly means a lot!
