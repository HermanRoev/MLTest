from torch.utils.data import DataLoader
from .coco import load_data

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loaders(train_data_dir, val_data_dir, train_ann_dir, val_ann_dir, batch_size):
    # Load the training and validation datasets
    train_data, val_data = load_data(train_data_dir, val_data_dir, train_ann_dir, val_ann_dir)

    # Create the data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_data_loader, val_data_loader