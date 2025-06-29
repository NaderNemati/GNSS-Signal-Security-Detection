# GNSS-Signal-Security-Detection

This study aims to explore and implement effective deep learning methodologies for detecting Global Navigation Satellite System (GNSS) signal threats, specifically focusing on jamming and spoofing attacks!
In this regard, two public datasets are utilized: an IQ-based spectrogram dataset for jamming classification and a tabular dataset containing satellite signal parameters for spoofing detection, both of which are leveraged. The objective is to develop robust classification pipelines for each threat modality. These implementations not only replicate state-of-the-art deep neural network models but also highlight the efficacy of data-driven approaches in enhancing the security of GNSS-dependent systems, such as maritime navigation platforms. 



### Raw IQ Dataset for GNSS Jamming Detection

Published by Swinney and Woods (2021) on Zenodo, this dataset consists of 120,000 spectrogram images derived from Short-Time Fourier Transform (STFT) of IQ samples. Each sample represents a 2D time-frequency plot, enabling visual discrimination between jamming types. The dataset is balanced across six classes (≈20,000 each):

    NoJam: Clean GNSS signals (baseline)

    SingleAM: Amplitude-modulated interference

    SingleFM: Frequency-modulated interference

    SingleChirp: Chirp jammers sweeping across frequencies

    Narrowband: Focused jamming on specific channels

    DME: Interference from Distance Measuring Equipment


### GPS Spoofing Detection Dataset (GPS Feature Map Dataset)


The spoofing dataset originates from GPS observations captured by an 8-channel receiver mounted on a mobile platform simulating a UAV. It includes ~510,530 signal instances, each described by 13 numerical features, including:

    PRN, Doppler offset, pseudorange

    Carrier phase, elevation angle, RX time

    C/N₀, packet count, and more

The dataset covers four classes: legitimate signals and three spoofing types (simplistic, intermediate, and sophisticated), representing varying spoofing complexity. Due to class imbalance, undersampling was used to prepare a balanced training subset.


## Repository Structure

📦 GNSS-Signal-Security-Detection


```bash
├── 📁 data/
│ ├── Raw_IQ_Dataset/ (Jamming Dataset from Zenodo)
│ └── GPS_Data_Simplified_2D_Feature_Map.xlsx (Spoofing Dataset)
├── 📁 models/
│ ├── cnn_jamming_model.py
│ └── dnn_spoofing_model.py
├── 📁 utils/
│ └── preprocessing.py
├── 📊 results/
├── 📜 README.md
└── 📄 requirements.txt

```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/GNSS-Signal-Security-Detection.git
cd GNSS-Signal-Security-Detection

```

2. Install dependencies:
   
```bash
pip install -r requirements.txt

```

## Run Pipelines

To run spoofing detection:

```bash
python spoofing_detection.py
```

## To run jamming detection:

```bash
python jamming_detection.py
```










## Spoofing Detection Pipeline

Dataset: Feature-mapped signals from a GPS receiver simulating UAV-based spoofing detection.

Implementation:

1-Load and clean tabular GPS observation data (.xlsx).

2-Apply label encoding and one-hot encoding to the output classes.

3-Normalize features with StandardScaler.

4-Train/test split with stratified sampling.

Model: DNN with 3 hidden layers and dropout regularization.

Loss function: Categorical crossentropy.

Optimizer: Nadam.

Evaluation: Accuracy, confusion matrix, and training history plots.

Output: Multi-class spoofing detection with high accuracy and generalization performance.


## Jamming Detection Pipeline

Dataset: Raw IQ samples converted to spectrograms (STFT images).

Implementation:

1-Read .mat IQ signal files.

2-Extract time-frequency spectrograms using scipy.signal.spectrogram().

3-Log-transform and resize spectrograms to 128x128 pixels.

4-Normalize and expand image dimensions.

5-Use image augmentation (ImageDataGenerator).

Model: CNN with 3 convolutional layers, batch normalization, max pooling, and dropout.

Optimizer: Nadam.

Training: Fit with early stopping and learning rate reduction.

Evaluation: Classification report, confusion matrix, and accuracy/loss curves.

Output: 6-class jamming classification (DME, NB, NoJam, AM, Chirp, FM) with strong visual features.




---

## References

- Swinney, D., & Woods, R. (2021). *GNSS Signal Jamming Dataset*. Zenodo. https://zenodo.org/record/4588474  
- Ranganathan, P. et al. (2023). *A Dataset for GPS Spoofing Detection on Unmanned Aerial System*. arXiv:2501.02352  
- GNSS Jamming and Spoofing Detection using Machine Learning and Computer Vision. GitHub Repository: https://github.com/alicmu2024/GNSS-Jamming-Detection...

