import torch
import torch.nn
from model import *

LAMBDA_C = 0.1
LAMBDA_I = 0.1


model = CycleGAN()

x = torch.randn(1, 24, 128)
y = torch.randn(1, 24, 128)

fake_x, fake_y, cycle_x, cycle_y, x_id, y_id, d_fake_x, d_fake_y, d_real_x, d_real_y = model(x,y)


real_label = torch.ones(d_fake_x.size())
fake_label = torch.zeros(d_fake_x.size())

# Train Generators
model.set_requires_grad([model.D_x, model.D_y], False)
# GAN LOSS

loss_G_x2y = F.binary_cross_entropy(d_fake_y, real_label)
loss_G_y2x = F.binary_cross_entropy(d_fake_x, real_label)
loss_gan = loss_G_x2y + loss_G_y2x

# Cycle LOSS

loss_cycle_x2y2x = F.l1_loss(cycle_x, x)
loss_cycle_y2x2x = F.l1_loss(cycle_y, y)
loss_cycle = loss_cycle_x2y2x + loss_cycle_y2x2x

# Identity LOSS

loss_identity_x2y = F.l1_loss(x_id, x)
loss_identity_y2x = F.l1_loss(y_id, y)
loss_identity = loss_identity_x2y + loss_identity_y2x

loss_total = loss_gan + LAMBDA_C * loss_cycle + LAMBDA_I * loss_identity

# Train Discriminators
model.set_requires_grad([model.D_x, model.D_y], True)

loss_D_fake_x = F.binary_cross_entropy(d_fake_x, fake_label)
loss_D_fake_y = F.binary_cross_entropy(d_fake_y, fake_label)
loss_D_real_x = F.binary_cross_entropy(d_real_x, real_label)
loss_D_real_y = F.binary_cross_entropy(d_real_y, real_label)

loss_dis = loss_D_fake_x + loss_D_fake_y + loss_D_real_x + loss_D_real_y






