import torch.nn as nn
from models import Generator, Discriminator
from config import config
import torch.optim as optim
from torch.autograd import Variable
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset


class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.loss = nn.MSELoss()

        self.generator.cuda()
        self.discriminator.cuda()
        self.loss.cuda()

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr, betas=(config.beta, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(config.beta, 0.999))

    def train(self, dataloader):
        noise = torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda()
        label_real = torch.FloatTensor(config.batch_size, 1).fill_(1).cuda()
        label_fake = torch.FloatTensor(config.batch_size, 1).fill_(0).cuda()

        for epoch in range(config.n_epoch):
            for i, (data, _) in enumerate(dataloader, 0):
                noise.data.normal_(0, 1)

                # Train Discriminator
                real = Variable(data.cuda())
                d_real = self.discriminator(real)

                fake = self.generator(noise)
                d_fake = self.discriminator(fake.detach())

                # d_real --> 1, d_fake --> 0
                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake)

                # update discriminator
                self.optimizer_d.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                d_fake = self.discriminator(fake)

                # d_fake --> 1
                g_loss = self.loss(d_fake, label_real)

                # update generator
                self.optimizer_g.zero_grad()
                g_loss.backward()
                self.optimizer_g.step()

                if i % config.log_iter == 0:
                    print("[Epoch {:03d}] ({}/{}) d_real: {}, d_fake: {}".format(epoch, i, len(dataloader),
                                                                                 d_real.mean(), d_fake.mean()))
                    vutils.save_image(fake.data,
                                      '{}/result_epoch_{:03d}_iter_{:05d}.png'.format(config.result_dir, epoch, i),
                                      normalize=True)

def main():
    dataset = dset.ImageFolder(config.dataset_dir, transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.n_cpu, pin_memory=True, drop_last=True)

    trainer = Trainer()
    trainer.train(dataloader)


if __name__ == '__main__':
    main()