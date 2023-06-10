import os
import json
from collections import OrderedDict
from torchvision import  transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, GTSRB, Food101, SUN397, EuroSAT, UCF101, StanfordCars, Flowers102, DTD, OxfordIIITPet
import numpy as np

from const import GTSRB_LABEL_MAP, IMAGENETNORMALIZE
'''
    function for loading datasets
    contains: 
        1. CIFAR-10
        2. CIFAR-100   
        3. SVHN
        4. GTSRB
        5. FOOD-101
        6. SUN-397
        7. EUROSAT
        8. UCF-101
        9. Stanford Cars
        10. FLOWERS-102
        11. DTD
        12. Oxford Pets
'''

def get_model(args):
    # network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(args.device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(args.device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    
    return network


# Imagenet Transform
def image_transform(args):
    normalize = transforms.Normalize(mean=IMAGENETNORMALIZE['mean'], std=IMAGENETNORMALIZE['std'])
    if args.prompt_method:
        
        if args.randomcrop and args.dataset=='cifar10':
            print('Using randomcrop\n')
            train_transform = transforms.Compose([
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
            ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform, test_transform, normalize


def get_torch_dataset(args):
    data_path = os.path.join(args.data, args.dataset)
    dataset = args.dataset
    train_transform, test_transform, normalize = image_transform(args)

    if dataset == "cifar10":
        full_data = CIFAR10(root = data_path, train = True, download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(CIFAR10(data_path, train=True, transform=train_transform, download=True), list(range(train_len)))
        val_set = Subset(CIFAR10(data_path, train=True, transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = CIFAR10(data_path, train=False, transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == "cifar100":
        full_data = CIFAR100(root = data_path, train = True, download = True)
        full_len = len(full_data)
        train_len = int(full_len * 0.9)
        train_set = Subset(CIFAR100(data_path, train=True, transform=train_transform, download=True), list(range(train_len)))
        val_set = Subset(CIFAR100(data_path, train=True, transform=test_transform, download=True), list(range(train_len, full_len)))
        test_set = CIFAR100(data_path, train=False, transform=test_transform, download=True)
        class_cnt = 100

    elif dataset == "svhn":
        full_data = SVHN(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(SVHN(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(SVHN(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = SVHN(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == "gtsrb":
        full_data = GTSRB(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(GTSRB(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(GTSRB(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = GTSRB(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 43

    elif dataset == 'food101':
        full_data = Food101(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(Food101(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(Food101(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = Food101(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 101

    elif dataset == 'sun397':
        full_data = SUN397(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(SUN397(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(SUN397(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = SUN397(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 397

    elif dataset == 'eurosat':
        full_data = EuroSAT(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(EuroSAT(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(EuroSAT(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = EuroSAT(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 10

    elif dataset == 'ucf101':
        full_data = UCF101(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(UCF101(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(UCF101(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = UCF101(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 101
    
    elif dataset == 'stanfordcars':
        full_data = StanfordCars(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(StanfordCars(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(StanfordCars(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = StanfordCars(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 196
    
    elif dataset == 'flowers102':
        full_data = Flowers102(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(Flowers102(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(Flowers102(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = Flowers102(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 102
    
    elif dataset == 'dtd':
        full_data = DTD(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(DTD(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(DTD(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = DTD(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 47

    elif dataset == 'oxfordpets':
        full_data = OxfordIIITPet(root = data_path, split = 'train', download = True)
        train_indices, val_indices = get_indices(full_data)
        train_set = Subset(OxfordIIITPet(data_path, split = 'train', transform=train_transform, download=True), train_indices)
        val_set = Subset(OxfordIIITPet(data_path, split = 'train', transform=test_transform, download=True), val_indices)
        test_set = OxfordIIITPet(data_path, split = 'test', transform=test_transform, download=True)
        class_cnt = 37

    else:
        raise NotImplementedError(f"{dataset} not supported")
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
    args.class_cnt = class_cnt
    args.normalize = normalize
    print(f'Dataset information: {dataset}\t {len(train_set)} images for training \t {len(val_set)} images for validation\t')
    print(f'{len(test_set)} images for testing\t')

    return train_loader, val_loader, test_loader


def get_indices(full_data):
    full_len = len(full_data)
    train_len = int(full_len * 0.9)
    indices = np.random.permutation(full_len)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    return train_indices, val_indices
