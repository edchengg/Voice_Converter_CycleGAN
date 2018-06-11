import torch
import torch.nn
from model.model import *

LAMBDA_C = 0.1
LAMBDA_I = 0.1


model = CycleGAN()


optimizer_G = torch.optim.Adam(model.G_params)
optimizer_D = torch.optim.Adam(model.D_params)


x = torch.randn(1, 24, 128) # (batch, channel, wT)
y = torch.randn(1, 24, 128)



optimizer_D.zero_grad()
optimizer_G.zero_grad()

fake_x, fake_y, cycle_x, cycle_y, x_id, y_id, d_fake_x, d_fake_y, d_real_x, d_real_y = model(x,y)


real_label = torch.ones(d_fake_x.size())
fake_label = torch.zeros(d_fake_x.size())

# Train Generators
model.train_G()
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

loss_total = loss_gan+ LAMBDA_C * loss_cycle + LAMBDA_I * loss_identity

loss_total.backward(retain_graph=True)
optimizer_G.step()

# Train Discriminators
model.train_D()

loss_D_fake_x = F.binary_cross_entropy(d_fake_x, fake_label)
loss_D_fake_y = F.binary_cross_entropy(d_fake_y, fake_label)
loss_D_real_x = F.binary_cross_entropy(d_real_x, real_label)
loss_D_real_y = F.binary_cross_entropy(d_real_y, real_label)

loss_dis = loss_D_fake_x + loss_D_fake_y + loss_D_real_x + loss_D_real_y
loss_dis.backward()
optimizer_D.step()






