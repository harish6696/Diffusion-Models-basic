import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

"""This code is based on the DDPM Paper: https://arxiv.org/abs/2006.11239"""

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device) #linearly increasing noise level for each time-step (no. of steps = 1000 by default) shape [1000]
        self.alpha = 1. - self.beta #shape [1000]
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #shape [1000]

    def prepare_noise_schedule(self):  #linear-scheduler 
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t): #Forward diffusion pass (directly generate the noised image at time t)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):  #Sample a timestep (step 3)
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    #Inference (Algo 2)
    def sample(self, model, n): 
        logging.info(f"Sampling {n} new images....")
        model.eval() #sets the model to evaluation mode, Specifically, layers like dropout layers or batch normalization layers may have different behavior during training and evaluation.
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) #initial image sampled from a normal distribution (pure noise) (shape: [n,3,64,64] for cifar10-64)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # loop starting from 1000 to 1 (by default)
                t = (torch.ones(n) * i).long().to(self.device) #creates [i,i, i ... i] (shape: [n,])
                predicted_noise = model(x, t)  #from the model, we predict the noise at time t. For n=8,i=1, t=[1,1,1,1,1,1,1,1]
                alpha = self.alpha[t][:, None, None, None]  #self.alpha[t=1] = [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999]], shape = [8]
                                                            #self.alpha[t][:,None,None,None]. shape =[8,1,1,1]       
                alpha_hat = self.alpha_hat[t][:, None, None, None] 
                beta = self.beta[t][:, None, None, None]            
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) #at the last step, we don't add noise
                """Denoising step: noise is removed sequentially from all the 'n' images"""
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                #shape of x = [8,3,64,64]       
        model.train()  #again set to training mode because sampling is called after every epoch in training loop
#        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8) #recover the original image
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader) #for batch size 4, l = 15000

    for epoch in range(args.epochs): #each epoch runs over the entire dataset
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader) #We get a batch of images from the dataloader. One batch has 15000 images. pbar has access to all the images in the dataset i.e. 60000
        for i, (images, _) in enumerate(pbar): #loop over 
            images = images.to(device)     #images hace shape [4,3,64,64] for batch size 4
            t = diffusion.sample_timesteps(images.shape[0]).to(device) #randomly sample 4 timesteps between 1 and 1000. t:(shape: [4,])
            x_t, noise = diffusion.noise_images(images, t) #for a given time t, we get the noised image (x_t) and the noise (Forward process), noise.shape = [4,3,64,64]
            predicted_noise = model(x_t, t) #shape of predicted_noise = [4,3,64,64]
            loss = mse(noise, predicted_noise) #scalar value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0]) #running inference to see how the quality of image improves with epochs
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg")) 
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"cifar10-64"  #training data has a total of 50,000 images and test data has 10,000 images
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
