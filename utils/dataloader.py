import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import config

def get_dataloader():
    dataset = dset.ImageFolder(root=config.DATA.dataroot,
                            transform=transforms.Compose([
                            transforms.Resize(config.DATA.image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            #transforms.RandomRotation((0,20)),
                            #transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

    dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=config.DATA.batch_size,
                                         shuffle=True, 
                                         num_workers=config.DATA.workers)
    return dataloader