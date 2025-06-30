<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/GNSS-Signal-Security-Detection/blob/main/images/jammingspoofing.png" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>






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






## Results Summary

This study evaluates two independent pipelines—**jamming** and **spoofing detection**—using deep learning models tailored to their respective dataset modalities.

### Jamming Detection (CNN on Spectrograms)

- **Test Accuracy:** `96.0%`  
- **Test Loss:** `0.1717`

| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| DME           | 1.00      | 1.00   | 1.00     | 250     |
| Narrowband    | 0.96      | 1.00   | 0.98     | 250     |
| NoJam         | 1.00      | 1.00   | 1.00     | 250     |
| SingleAM      | 0.80      | 0.99   | 0.88     | 250     |
| SingleChirp   | 1.00      | 0.96   | 0.98     | 250     |
| SingleFM      | 0.99      | 0.75   | 0.85     | 250     |

- **Overall Metrics:**
  - Accuracy: **96.0%**
  - Macro F1-score: **0.96**
  - Weighted F1-score: **0.96**

<sub>Model: CNN trained on 128×128 log-scaled spectrograms with data augmentation, dropout regularization, early stopping, and learning rate scheduling.</sub>



<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/GNSS-Signal-Security-Detection/blob/main/images/Jamming2.png" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>


In the confusion matrix of jamming detection, the jamming classification model performs well, correctly identifying all 250 instances for DME, NB, and NoJam. Similarly, the SingleChirp and SingleAM classes achieved 96% and 99.2% accuracy, respectively. The model, however, shows noticeable confusion between single FM and single AM samples, with 63 out of 250 single FM samples misclassified as single AM, resulting in a class-wise accuracy of 74.8%. As a result, these two classes may share spectral characteristics that need to be refined or enhanced further.

---

### ✅ Spoofing Detection (DNN on Feature Map)

- **Test Accuracy:** `95.72%`  
- **Test Loss:** `0.1256`  
- **Final Epoch:** `100`

<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/GNSS-Signal-Security-Detection/blob/main/images/spoofing1.png" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>

The confusion matrix for the spoofing detection model shows strong classification performance across the four classes of spoofing. The most accurate classification is in Class 0 (legitimate signals), with 78,029 out of its total predictions correctly classified and relatively low misclassification rates. There are 5,708 correct predictions in Class 3 (sophisticated spoofing) and minimal confusion with other classes. There is some overlap between Class 1 (simplistic spoofing) and Class 2 (intermediate spoofing), particularly with Class 2 misclassified as Class 0 in 2,572 cases and as Class 1 in 133 cases. As a result of this confusion between spoofing classes, the model demonstrates both benign and complex spoofing detection capability with notably high accuracy.



## Repository Structure


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
- GNSS Jamming and Spoofing Detection using Machine Learning and Computer Vision. GitHub Repository: https://github.com/alicmu2024/GNSS-Jamming-Detection

