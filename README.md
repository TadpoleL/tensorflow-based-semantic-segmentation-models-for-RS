# TensorFlow-based Semantic Segmentation Models for Remote Sensing

This repository contains various semantic segmentation models implemented using TensorFlow 2.14. The models included are:

- DeepLab
- HRNet
- LSTM
- MACUNet
- ResUNet
- SegNet
- UNet
- UNet+++
- U-HRNet

## Features

- **Multispectral Image Support**: Both training and prediction scripts support multispectral image data.
- **State-of-the-art Models**: Includes popular and state-of-the-art semantic segmentation models tailored for remote sensing applications.

## Getting Started

### Prerequisites

- TensorFlow 2.14
- Python 3.x
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/TadpoleL/tensorflow-based-semantic-segmentation-models-for-RS.git
    cd tensorflow-based-semantic-segmentation-models-for-RS
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Training

To train a model, run the training script with the appropriate configuration. 

### Prediction

To perform prediction using a trained model, use the prediction script. 

## Models Description

- **DeepLab**: A deep learning model for semantic image segmentation, offering several variants such as DeepLabV3 and DeepLabV3+.
- **HRNet**: High-Resolution Network, which maintains high-resolution representations through the whole process.
- **LSTM**: Long Short-Term Memory networks, adapted for segmentation tasks.
- **MACUNet**: A variation of UNet with additional attention mechanisms.
- **ResUNet**: UNet with residual connections.
- **SegNet**: A deep convolutional encoder-decoder architecture for image segmentation.
- **UNet**: A convolutional network architecture for fast and precise segmentation of images.
- **UNet+++**: An advanced version of UNet with nested and dense skip connections.
- **U-HRNet**: Combines the strengths of UNet and HRNet for improved segmentation performance.

## Contributing

Contributions are welcome! Please submit pull requests or open issues to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
