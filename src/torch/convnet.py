import torch
import torch.nn as nn
from libs.utils import datasets, dataset_processors as dp

torch.set_printoptions(precision=2)
# Define the Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Assuming output size for MNIST classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x

# Create an instance of the CNN
model = CNN()

# Load a sample image from the MNIST dataset (assuming you have already loaded the dataset)
# Here 'sample_input' should be a tensor of shape (batch_size, channels, height, width)
# For MNIST dataset, it will be (1, 1, 28, 28)

(x_train, y_train), (x_test, y_test) = datasets.load_mnist()
x_train = torch.from_numpy(x_train).float()

sample_x = torch.reshape(x_train[0], (1, 1, 28, 28))
sample_input = torch.randn(1, 1, 28, 28)  # Example input

# Make a prediction
output = model(sample_input)
output2 = model(sample_x)

print("Output shape:", output.shape)
print("Output:", output)
