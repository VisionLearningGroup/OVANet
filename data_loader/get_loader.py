from .mydataset import ImageFolder, ImageFolder_aug
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, test_loader, target_folder_train



def get_loader_label(source_path, target_path, target_path_label, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    target_folder_label = ImageFolder(os.path.join(target_path_label),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    target_loader_label = torch.utils.data.DataLoader(
        target_folder_label,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, target_loader_label, test_loader, target_folder_train



def get_loader_target_aug(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder_aug(os.path.join(target_path),
                                      transform=transforms[target_path], transform_aug=transforms["aug"],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, test_loader, target_folder_train



