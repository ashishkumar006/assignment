import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define the same transform used in training
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate images by ±10 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load a sample of the MNIST dataset with transformations
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Function to display images
def show_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get a few samples
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Adjust as needed
images = []
labels = []

for idx in sample_indices:
    img, label = dataset[idx]
    images.append(img)
    labels.append(label)

# Show the images
show_images(images, labels)
