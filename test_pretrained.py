import torch
from torch import hub
import matplotlib.pyplot as plt
import torchvision
model = hub.load(
    'facebookresearch/pytorch_gan_zoo:master',
    'PGAN', #
    useGPU = True,
    pretrained=True)

BATCH_SIZE = 1

inputRandom, randomLabels = model.buildNoiseData(BATCH_SIZE)

G = model.netG()
D = model.netD()

generated_images = model.test(inputRandom)
plt.imsave('test.png', torchvision.utils.make_grid(generated_images).permute(1, 2, 0).cpu().numpy())
