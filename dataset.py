import os
import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=0
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std

def get_data_location():
    """
    Find the location of the dataset
    """

    if os.path.exists("/mnt/d/DeepLearning/SelfDrivingCar/data"):
        data_folder = "/mnt/d/DeepLearning/SelfDrivingCar/data"
    else:
        raise IOError("Please download the dataset first")

    return data_folder

class CULaneDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, mask_root, file_list, transform=None, target_transform=None):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.transform = transform
        self.target_transform = target_transform
        self.file_list = []  # File chứa đường dẫn ảnh và mask

        # Đọc file chứa danh sách các ảnh và mask
        with open(file_list, "r") as f:
            self.file_list = f.readlines()  # Lưu tất cả các dòng trong file vào list

        self.samples = []
        for line in self.file_list:
            parts = line.strip().split()  # Tách dòng thành các phần (đường dẫn ảnh và mask)
            if len(parts) < 2:
                continue

            # Lấy đường dẫn ảnh và mask tương đối
            image_rel = parts[0].lstrip('/')  # Loại bỏ dấu '/' ở đầu đường dẫn ảnh
            mask_rel = parts[1].lstrip('/')  # Loại bỏ dấu '/' ở đầu đường dẫn mask

            # Gắn path đầy đủ cho ảnh
            image_path = self.image_root / image_rel

            # Đảm bảo rằng đường dẫn mask đầy đủ
            # Thêm một tầng "laneseg_label_w16" vào đường dẫn mask
            mask_path = self.mask_root / "laneseg_label_w16" / Path(mask_rel).relative_to("laneseg_label_w16")

            # Nếu không tồn tại ảnh, thử bổ sung thêm một tầng thư mục cho ảnh
            if not image_path.exists():
                # Thêm folder con vào đường dẫn ảnh nếu không tìm thấy
                image_path = self.image_root / Path(image_rel).parts[0] / image_rel

            # Thêm path ảnh và mask vào danh sách mẫu
            self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        # Mở ảnh và mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        # Áp dụng các phép biến đổi (nếu có)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use num_workers=1. 
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location()) / "CULane"
    image_root = base_path
    mask_root = base_path / "laneseg_label_w16"
    list_path = base_path / "list"

    # Define transforms
    input_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    def make_loader(file_list, shuffle=True):
        dataset = CULaneDataset(
            image_root=image_root,
            mask_root=mask_root,
            file_list=file_list,
            transform=input_transform,
            target_transform=target_transform
        )
        if limit > 0:
            dataset = torch.utils.data.Subset(dataset, range(limit))
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    data_loaders["train"] = make_loader(list_path / "train_gt.txt", shuffle=True)
    data_loaders["valid"] = make_loader(list_path / "val_gt.txt", shuffle=False)
    data_loaders["test"]  = make_loader(list_path / "test.txt", shuffle=False)

    # Compute mean and std of the dataset
    # mean, std = compute_mean_and_std()
    # print(f"Dataset mean: {mean}, std: {std}")

    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # data_transforms = {
    #     "train": transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
    #         transforms.RandomRotation(degrees=10),                      # Xoay ảnh ±10 độ
    #         transforms.RandomHorizontalFlip(),                          # Lật ngẫu nhiên theo chiều ngang
    #         transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Dịch chuyển ảnh ±5% theo cả chiều ngang và dọc
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2),       # Điều chỉnh độ sáng và độ tương phản
    #         transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Biến dạng phối cảnh nhẹ
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ]),
    #     "valid": transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ]),
    #     "test": transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ]),
    # }

    # # Create train and validation datasets
    # train_data = datasets.ImageFolder(
    #     base_path / "train",
    #     # Add the appropriate transform that you defined in
    #     # the data_transforms dictionary
    #     transform=data_transforms['train']
    # )
    # # The validation dataset is a split from the train_one_epoch dataset, so we read
    # # from the same folder, but we apply the transforms for validation
    # valid_data = datasets.ImageFolder(
    #     base_path / "train",
    #     # Add the appropriate transform that you defined in
    #     # the data_transforms dictionary
    #     transform=data_transforms['valid']
    # )

    # # obtain training indices that will be used for validation
    # n_tot = len(train_data)
    # indices = torch.randperm(n_tot)

    # # If requested, limit the number of data points to consider
    # if limit > 0:
    #     indices = indices[:limit]
    #     n_tot = limit

    # split = int(math.ceil(valid_size * n_tot))
    # train_idx, valid_idx = indices[split:], indices[:split]

    # # define samplers for obtaining training and validation batches
    # train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    # valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx)

    # # prepare data loaders
    # data_loaders["train"] = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=batch_size,
    #     sampler=train_sampler,
    #     num_workers=num_workers,
    # )
    # data_loaders["valid"] = torch.utils.data.DataLoader(
    #     valid_data,
    #     batch_size=batch_size,
    #     sampler=valid_sampler,
    #     num_workers=num_workers,
    # )

    # # Now create the test data loader
    # test_data = datasets.ImageFolder(
    #     base_path / "test",
    #     transform=data_transforms['test'],
    # )

    # if limit > 0:
    #     indices = torch.arange(limit)
    #     test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    # else:
    #     test_sampler = None

    # data_loaders["test"] = torch.utils.data.DataLoader(
    #     test_data, 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=num_workers
    # )

    return data_loaders

def visualize_sample_from_loader(data_loader, num_samples=5):
    """
    Hiển thị một vài ảnh + mask từ một DataLoader (train/valid/test)
    """
    for images, masks in data_loader:
        for i in range(min(num_samples, len(images))):
            image = images[i]
            mask = masks[i]

            # Convert tensor về numpy
            img_np = F.to_pil_image(image).convert("RGB")
            mask_np = mask.numpy()

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title("Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title("Lane Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        break  # chỉ lấy 1 batch