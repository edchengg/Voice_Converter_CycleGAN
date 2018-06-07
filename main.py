import torch
import torch.nn
from model import *

LAMBDA_C = 0.1
LAMBDA_I = 0.1


CycleGAN = CycleGAN()

x = torch.randn(1, 24, 128)
y = torch.randn(1, 24, 128)

x2y, y2x, y_fake, x_fake, dis_fake_y, dis_fake_x, dis_real_y, dis_real_x,y_ident,x_ident = CycleGAN(x,y)


# Gan loss
fool_dis_loss = F.mse_loss(dis_fake_x, torch.ones(dis_fake_x.size())) + F.mse_loss(dis_fake_y, torch.ones(dis_fake_y.size()))

# Cycle consistent loss
consis_loss = F.l1_loss(x_fake, x) + F.l1_loss(y_fake, y)
# Identity mapping loss
ident_loss = F.l1_loss(x_ident, x) + F.l1_loss(y_ident, y)

gan_loss = fool_dis_loss + LAMBDA_C * consis_loss + LAMBDA_I * ident_loss


# Discriminator loss
d_real_loss = F.mse_loss(dis_real_x, torch.ones(dis_real_x.size())) + \
               F.mse_loss(dis_real_y, torch.ones(dis_real_y.size()))

d_fake_loss = F.mse_loss(dis_fake_x, torch.zeros(dis_fake_x.size())) + \
              F.mse_loss(dis_fake_y, torch.zeros(dis_fake_y.size()))

discriminator_loss = d_real_loss + d_fake_loss






