from dataLoaders import *
import os
import argparse
import random
import ipdb

abs_path = '/home/ferles/medusa/src/'
global_seed = 1
torch.backends.cudnn.deterministic = True
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)


def _create_fgsm_loader(val_loader, device):

    sample, gts = next(iter(val_loader))
    sizes = sample.size()
    len_ = 0
    ood_data_x = torch.zeros(size=(val_loader.__len__()*val_loader.batch_size, sizes[1], sizes[2], sizes[3]))
    ood_data_y = torch.zeros(val_loader.__len__()*val_loader.batch_size)
    fgsm_step = 0.1
    criterion = nn.CrossEntropyLoss()
    for index, data in enumerate(val_loader):

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        input_var = torch.autograd.Variable(images, requires_grad=True)
        input_var = input_var.to(device)
        output = model(input_var)
        if len(labels.size()) > 1:
            labels = torch.argmax(labels, dim=1)
        loss = criterion(output, labels)
        loss.backward()

        sign_data_grad = input_var.grad.data.sign()
        perturbed_image = input_var + fgsm_step*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        ood_data_x[index*val_loader.batch_size:index*val_loader.batch_size+images.size(0)] = perturbed_image
        ood_data_y[index*val_loader.batch_size:index*val_loader.batch_size+images.size(0)] = labels
        len_ += images.size(0)

    ood_data_x, ood_data_y = ood_data_x[:len_], ood_data_y[:len_]
    fgsm_dataset = TensorDataset(ood_data_x, ood_data_y)
    fgsm_loader = DataLoader(fgsm_dataset, batch_size=val_loader.batch_size)

    return fgsm_loader



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='fgsm')
    parser.add_argument('--device', '--dv', type=int, default=0, required=False)
    args = parser.parse_args()

    dataloader_1_isic = imageNetLoader(batch_size=1)
    dataloader_1 = imageNetLoader(batch_size=1)
    dataloader_10_isic = imageNetLoader(batch_size=10)
    dataloader_10 = imageNetLoader(batch_size=10)
    dataloader_32_isic = imageNetLoader(batch_size=32)
    dataloader_32 = imageNetLoader(batch_size=32)

    if not os.path.exists('imageNetFGSM/'):
        os.system('mkdir imageNetFGSM/')
    if not os.path.exists('imageNetVal/'):
        os.system('mkdir imageNetFGSM/')

    fgsm_dataloader_1_isic = _create_fgsm_loader(dataloader_1_isic, args.device)
    fgsm_dataloader_1 = _create_fgsm_loader(dataloader_1, args.device)
    fgsm_dataloader_10_isic = _create_fgsm_loader(dataloader_10_isic, args.device)
    fgsm_dataloader_10 = _create_fgsm_loader(dataloader_10, args.device)
    fgsm_dataloader_32_isic = _create_fgsm_loader(dataloader_32_isic, args.device)
    fgsm_dataloader_32 = _create_fgsm_loader(dataloader_32, args.device)