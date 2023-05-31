from torchvision.datasets import CocoDetection
from transforms.transforms import BoxCoordinates, Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize


# Define the transformation
def transform(image, target):
    transform = Compose([
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        BoxCoordinates(),
    ])
    return transform(image, target)


# Load the training and validation datasets
def load_data(train_data_dir, val_data_dir, train_ann_dir, val_ann_dir):
    # Load the training data
    train_data = CocoDetection(root=train_data_dir, annFile=train_ann_dir, transform=transform)

    # Load the validation data
    val_data = CocoDetection(root=val_data_dir, annFile=val_ann_dir, transform=transform)

    return train_data, val_data
