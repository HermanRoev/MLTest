import torch
import torch.optim as optim
from data.dataloader import get_data_loaders
from models.model import get_model
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 10
batch_size = 4
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005


data_dir = r"C:\Users\Herma\PycharmProjects\data"  # Replace with the actual path to the data directory

train_data_dir = os.path.join(data_dir, "train2017")
val_data_dir = os.path.join(data_dir, "val2017")
train_ann_dir = os.path.join(data_dir, "annotations/instances_train2017.json")
val_ann_dir = os.path.join(data_dir, "annotations/instances_val2017.json")

# Get the data loaders
train_data_loader, val_data_loader = get_data_loaders(train_data_dir, val_data_dir, train_ann_dir, val_ann_dir, batch_size)


num_classes = 91  # Number of classes in COCO dataset
model = get_model(num_classes)
model.to(device)


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, targets in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    # Update the learning rate
    lr_scheduler.step()

    # Print the training loss for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_data_loader)}")

torch.save(model.state_dict(), "model.pth")
