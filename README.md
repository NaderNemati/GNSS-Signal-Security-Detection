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



