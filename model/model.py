import os
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from utils import *
from preprocess import *
import numpy as np

def padding_same(input_size, kernel, stride):
    p = stride * (input_size - 1) - input_size + kernel
    p = p // 2
    return p


def padding_half(input_size, kernel, stride):
    p1 = ((input_size / 2 - 1) * stride + kernel - input_size) / 2
    p2 = ((input_size / 2) * stride + kernel - input_size) / 2
    if p1 == input_size / 2:
        return p1
    else:
        return p2



class GatedCNN1d(nn.Module):
    def __init__(self,
                 in_chs,
                 out_chs,
                 kernel,
                 stride,
                 padding,
                 ins_norm=True,
                 shuffle=False):
        super(GatedCNN1d, self).__init__()

        self.conv_0 = nn.Conv1d(in_chs, out_chs, kernel_size=kernel, stride=stride, padding=padding)
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1))
        self.conv_gate_0 = nn.Conv1d(in_chs, out_chs, kernel_size=kernel, stride=stride, padding=padding)
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1))

        # Use instance normalization indicator
        self.ins_norm = ins_norm
        if ins_norm:
            self.conv1d_norm = nn.InstanceNorm1d(out_chs, eps=1e-6, affine=True)
        self.shuffle = shuffle

    def forward(self, x):
        # x: (batch_size, Cin, W)

        A = self.conv_0(x)
        Wout = A.size(2)
        A += self.b_0.repeat(1, 1, Wout)
        B = self.conv_gate_0(x)
        B += self.c_0.repeat(1, 1, Wout)

        if self.shuffle:
            A = self.pixel_shuffle(A)
            B = self.pixel_shuffle(B)

        if self.ins_norm:
            A = self.conv1d_norm(A)
            B = self.conv1d_norm(B)

        h = A * F.sigmoid(B)
        return h

    def pixel_shuffle(self, x, shuffle_size=2):
        n = x.size(0)
        c = x.size(1)
        w = x.size(2)
        out_c = c // shuffle_size
        out_w = w * shuffle_size

        output = x.reshape(n, out_c, out_w)
        return output


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_chs,
                 out_chs,
                 out_chs2,
                 kernel1,
                 kernel2,
                 stride1,
                 stride2,
                 padding,
                 ins_norm):
        super(ResidualBlock, self).__init__()

        self.GLU = GatedCNN1d(in_chs, out_chs, kernel1, stride1, padding, ins_norm)
        self.conv1d = nn.Conv1d(out_chs, out_chs2, kernel2, stride2, padding)
        self.conv1d_norm = nn.InstanceNorm1d(out_chs2, eps=1e-6, affine=True)

    def forward(self, x):
        out = self.GLU(x)
        out = self.conv1d(out)
        out = self.conv1d_norm(out)
        out += x
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.input_layer = nn.Conv1d(24, 128, kernel_size=15, stride=1, padding=padding_same(128, 15, 1))
        self.input_layer_gates = nn.Conv1d(24, 128, kernel_size=15, stride=1, padding=padding_same(128, 15, 1))

        self.down_sample = nn.Sequential(
            GatedCNN1d(128, 256, kernel=5, stride=2, padding=padding_half(128, 5, 2), ins_norm=True),
            GatedCNN1d(256, 512, kernel=5, stride=2, padding=padding_half(64, 5, 2), ins_norm=True)
        )
        self.residual_blocks = self.build_residual_blocks(6)

        self.up_sample = nn.Sequential(
            GatedCNN1d(512, 1024, kernel=5, stride=1, padding=padding_same(32, 5, 1), ins_norm=True, shuffle=True),
            GatedCNN1d(512, 512, kernel=5, stride=1, padding=padding_same(64, 5, 1), ins_norm=True, shuffle=True)
        )

        self.output_layer = nn.Conv1d(256, 24, kernel_size=15, stride=1, padding=padding_same(128, 15, 1))

    def build_residual_blocks(self, num_blocks):
        conv_blocks = []
        conv_blocks += [ResidualBlock(in_chs=512, out_chs=1024, out_chs2=512, kernel1=3,
                                      kernel2=3, stride1=1, stride2=1, padding=padding_same(32, 3, 1),
                                      ins_norm=True)
                        for _ in range(num_blocks)]

        return nn.Sequential(*conv_blocks)

    def forward(self, inputs):
        A = self.input_layer(inputs)
        B = self.input_layer_gates(inputs)
        H = A * F.sigmoid(B)
        H = self.down_sample(H)
        H = self.residual_blocks(H)
        H = self.up_sample(H)
        H = self.output_layer(H)
        return H




class Downsample2d(nn.Module):
    def __init__(self,
                 in_chs,
                 out_chs,
                 kernel,
                 stride,
                 padding):
        super(Downsample2d, self).__init__()

        self.conv_0 = nn.Conv2d(in_chs, out_chs, kernel_size=kernel, stride=stride, padding=padding)
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(in_chs, out_chs, kernel_size=kernel, stride=stride, padding=padding)
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.ins_norm = nn.InstanceNorm2d(out_chs, eps=1e-6, affine=True)

    def forward(self, x):
        A = self.conv_0(x)
        Hout, Wout = A.size(2), A.size(3)
        A += self.b_0.repeat(1, 1, Hout, Wout)
        B = self.conv_gate_0(x)
        B += self.c_0.repeat(1, 1, Hout, Wout)
        A = self.ins_norm(A)
        B = self.ins_norm(B)
        H = A * F.sigmoid(B)
        return H


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_layer = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.input_layer_gates = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))

        self.down_sample_1 = Downsample2d(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.down_sample_2 = Downsample2d(256, 512, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.down_sample_3 = Downsample2d(512, 1024, kernel=(3, 3), stride=(1, 2), padding=(1, 1))

        self.fc = nn.Linear(1024 * 6 * 8, 6 * 8)

    def forward(self, x):
        batch = x.size(0)
        #print(x.size())
        A = self.input_layer(x)
        B = self.input_layer_gates(x)
        H = A * F.sigmoid(B)
        #print(H.size())
        H = self.down_sample_1(H)
        #print(H.size())
        H = self.down_sample_2(H)
        #print(H.size())
        H = self.down_sample_3(H)
        #print(H.size())
        H = H.view(batch, -1)
        out = self.fc(H)
        #print(out.size())

        #out = F.sigmoid(out)
        return out



class CycleGAN(nn.Module):
    def __init__(self, cuda=False):
        super(CycleGAN, self).__init__()

        self.G_x2y = Generator()
        self.G_y2x = Generator()

        self.D_x = Discriminator()
        self.D_y = Discriminator()

        self.G_params = chain(self.G_x2y.parameters(), self.G_y2x.parameters())
        self.D_params = chain(self.D_x.parameters(), self.D_y.parameters())

        self.fake_x_buffer = ReplayBuffer()
        self.fake_y_buffer = ReplayBuffer()

        self.CUDA = cuda
        self.set_parm_()

    def forward(self, x, y):
        fake_y = self.G_x2y(x)
        fake_x = self.G_y2x(y)

        cycle_y = self.G_x2y(fake_x)
        cycle_x = self.G_y2x(fake_y)

        y_id = self.G_x2y(y)
        x_id = self.G_y2x(x)

        # Buffer operation(save fake to buffer and randomly select them)
        # fake_x_ = self.fake_x_buffer.push_and_pop(fake_x)
        # fake_y_ = self.fake_y_buffer.push_and_pop(fake_y)

        # trans x y to 2d
        fake_x_ = fake_x.unsqueeze(1)
        fake_y_ = fake_y.unsqueeze(1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        d_fake_x = self.D_x(fake_x_)
        d_fake_y = self.D_x(fake_y_)

        d_real_x = self.D_x(x)
        d_real_y = self.D_y(y)

        return fake_x,fake_y, cycle_x, cycle_y, x_id, y_id, d_fake_x, d_fake_y, d_real_x, d_real_y

    def set_parm_(self):
        self.sampling_rate = 16000
        self.frame_period = 5.0
        self.num_mcep = 24
        data_logf0 = np.load('model/sf1_tf2/logf0s_normalization.npz')
        data_mcep = np.load('model/sf1_tf2/mcep_normalization.npz')
        self.log_f0s_mean_A = data_logf0['mean_A']
        self.log_f0s_std_A = data_logf0['std_A']
        self.log_f0s_mean_B = data_logf0['mean_B']
        self.log_f0s_std_B = data_logf0['std_B']

        self.coded_sps_A_mean = data_mcep['mean_A']
        self.coded_sps_A_std = data_mcep['std_A']
        self.coded_sps_B_mean = data_mcep['mean_B']
        self.coded_sps_B_std = data_mcep['std_B']
        self.validation_A_dir = 'evaluation_all/SF1'
        self.validation_B_dir = 'evaluation_all/TF2'
        self.validation_A_output_dir = 'validation_output/converted_A'
        self.validation_B_output_dir = 'validation_output/converted_B'

    def trans_audio(self):

        self.G_x2y.eval()
        self.G_y2x.eval()

        for file in os.listdir(self.validation_A_dir):
            filepath = os.path.join(self.validation_A_dir, file)
            wav, _ = librosa.load(filepath, sr=self.sampling_rate, mono=True)
            wav = wav_padding(wav=wav, sr=self.sampling_rate, frame_period=self.frame_period, multiple=4)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=self.sampling_rate, frame_period=self.frame_period)
            f0_converted = pitch_conversion(f0=f0, mean_log_src=self.log_f0s_mean_A, std_log_src=self.log_f0s_std_A,
                                            mean_log_target=self.log_f0s_mean_B, std_log_target=self.log_f0s_std_B)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=self.sampling_rate, dim=self.num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed - self.coded_sps_A_mean) / self.coded_sps_A_std
            x = torch.Tensor(np.array([coded_sp_norm]))
            x = x.cuda() if self.CUDA else x
            coded_sp_converted_norm = self.G_x2y(x)
            if self.CUDA:
                coded_sp_converted_norm = coded_sp_converted_norm.cpu()
            coded_sp_converted_norm = coded_sp_converted_norm.detach().numpy()
            coded_sp_converted = coded_sp_converted_norm * self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.reshape(self.num_mcep, -1)
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=self.sampling_rate)
            wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap,
                                                     fs=self.sampling_rate,
                                                     frame_period=self.frame_period)
            librosa.output.write_wav(os.path.join(self.validation_A_output_dir, os.path.basename(file)), wav_transformed,
                                     self.sampling_rate)

        for file in os.listdir(self.validation_B_dir):
            filepath = os.path.join(self.validation_B_dir, file)
            wav, _ = librosa.load(filepath, sr=self.sampling_rate, mono=True)
            wav = wav_padding(wav=wav, sr=self.sampling_rate, frame_period=self.frame_period, multiple=4)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=self.sampling_rate, frame_period=self.frame_period)
            f0_converted = pitch_conversion(f0=f0, mean_log_src=self.log_f0s_mean_B, std_log_src=self.log_f0s_std_B,
                                            mean_log_target=self.log_f0s_mean_A, std_log_target=self.log_f0s_std_A)
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=self.sampling_rate, dim=self.num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed - self.coded_sps_B_mean) / self.coded_sps_B_std
            y = torch.Tensor(np.array([coded_sp_norm]))
            y = y.cuda() if self.CUDA else y
            coded_sp_converted_norm = self.G_y2x(y)
            if self.CUDA:
                coded_sp_converted_norm = coded_sp_converted_norm.cpu()
            coded_sp_converted_norm = coded_sp_converted_norm.detach().numpy()
            coded_sp_converted = coded_sp_converted_norm * self.coded_sps_A_std + self.coded_sps_A_mean
            coded_sp_converted = coded_sp_converted.reshape(self.num_mcep, -1)
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=self.sampling_rate)
            wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap,
                                                     fs=self.sampling_rate, frame_period=self.frame_period)
            librosa.output.write_wav(os.path.join(self.validation_B_output_dir, os.path.basename(file)), wav_transformed,
                                     self.sampling_rate)
        self.G_x2y.train()
        self.G_y2x.train()





