import torchvision.transforms as transforms
from data_loader.get_loader import get_loader, get_loader_label


def get_dataloaders(source_data, target_data, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms, batch_size=conf.data.dataloader.batch_size, return_id=True,
                      balanced=conf.data.dataloader.class_balance)


def get_dataloaders_label(source_data, target_data, target_data_label, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_label(source_data, target_data, target_data_label, evaluation_data,
                      data_transforms, batch_size=conf.data.dataloader.batch_size, return_id=True,
                      balanced=conf.data.dataloader.class_balance)