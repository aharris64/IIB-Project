import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(root_folder, dataset, batch_size):
    """
    Loads training, validation, and test datasets using ImageFolder and
    returns their corresponding DataLoaders
    """

    root = os.path.join(root_folder, dataset)

    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")
    test_dir  = os.path.join(root, "test")


    transform = transforms.Compose([
        transforms.ToTensor(),

        # Normalise using ImageNet statisticss (for pretrained models)
        transforms.Normalize( 
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds   = datasets.ImageFolder(val_dir, transform=transform)
    test_ds  = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print("Class mapping:", train_ds.class_to_idx)
    print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))

    return train_loader, val_loader, test_loader