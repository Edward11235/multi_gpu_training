import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import os
import time

# Define the GPU selection here
GPU_SELECTION = 'all'  # options: 'all', '0', '1'


def main():
    if GPU_SELECTION == 'all':
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    
    print("Running on {} GPU(s)".format(num_gpus))
    
    start = time.time()
    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(num_gpus, GPU_SELECTION,))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_SELECTION
        train(0, num_gpus, GPU_SELECTION)
        
    print("Training complete. Total time taken: {} seconds".format(time.time()-start))


def train(gpu, num_gpus, gpu_selection='all'):
    rank = gpu
    
    if gpu_selection == 'all':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', init_method='env://', world_size=num_gpus, rank=rank)
    
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu]) if gpu_selection == 'all' else model
    
    # Print the GPU name
    print(f"GPU{gpu} ({torch.cuda.get_device_name(gpu)}) is used for training")

    # Rest of the code...

class ConvNet(nn.Module):
    # The rest of the ConvNet class definition goes here...
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(4*4*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
if __name__=="__main__":
    main()
