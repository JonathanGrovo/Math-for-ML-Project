# Neural Network Comparator

This project demonstrates two approaches to classifying handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/):
1. A simple Convolutional Neural Network (CNN)
2. A basic Feedforward Neural Network (FNN)

Both models are trained and evaluated on the MNIST dataset, and their performance is compared via:
- Training/Validation loss curves
- Validation accuracy curves
- A confusion matrix on the test set

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/) (with GPU support if available)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)

Make sure you have a GPU-compatible version of PyTorch if you plan to use a CUDA-enabled device.

## How It Works

1. **Data Loading and Preprocessing**  
   - The MNIST dataset is downloaded automatically (if not present) and split into training, validation, and test sets.
   - The images are transformed (normalized) for better training stability.

2. **Model Architectures**  
   - **SimpleCNN**: A simple convolutional model with two convolutional layers, max-pooling, dropout, and two fully connected layers.
   - **FeedforwardNN**: A three-layer fully connected network using ReLU activations.

3. **Training**  
   - Each model is trained for a specified number of epochs (default: 5) with an Adam optimizer and Cross-Entropy loss.
   - The script logs the training and validation losses, as well as validation accuracy after each epoch.

4. **Evaluation**  
   - A confusion matrix is computed on the test set for both models.
   - Training/validation loss and validation accuracy curves are plotted for visual inspection of training progress.

## Usage

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib scikit-learn
   ```
3. Run the script:
   ```bash
   python main.py
   ```
   (Assuming the file is named `main.py`.)

During execution, you will see:
- Training logs for both CNN and FNN
- Confusion matrices for each model’s predictions on the test data
- Plots of training and validation losses, as well as validation accuracies

## Customization

- **Hyperparameters**: You can adjust `learning_rate`, `num_epochs`, `batch_size`, and `validation_split` in the script to suit your needs.
- **Data Location**: By default, MNIST data is downloaded to `./data`. Modify the `root` parameter in `datasets.MNIST` calls if you want to change the download or storage path.

## License

This project is provided as-is without any warranty. Feel free to modify and distribute as needed.
