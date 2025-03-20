import os

import torch
from torch import nn
import math
import torch.nn.functional as F
from math import sqrt
import numpy as np

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        # '../ltsm/checkpoints/lotsa_uea_ucr_bert_d1024_l8_p16_n512_weight_p50_wo_revin_full-epoch=25.ckpt'
        # 从configs.ckpt_path的字符串中获取patch_len，规则为文件名中第一个_p后的数字
        patch_len = int(configs.ckpt_path.split('_p')[1].split('_')[0])
        d_model = int(configs.ckpt_path.split('_d')[1].split('_')[0])
        layers = int(configs.ckpt_path.split('_l')[1].split('_')[0])
        d_ff = d_model * 4 if d_model == 768 or (d_model == 1024 and layers == 24) else d_model * 2
        n_heads = 12 if d_model == 768 else 8
        n_heads = 16 if d_model == 1024 and layers == 24 else n_heads
        print("patch_len:", patch_len)

        stride = None
        enc_in = 1,
        # d_model = 1024
        # d_ff = 2048
        # layers = 8
        # n_heads = 8
        dropout = 0.1
        activation = 'gelu'
        factor = 1
        position_embedding = False

        # d_model = 256
        # d_ff = 1024
        # layers = 3

        self.task_name = configs.task_name

        self.patch_len = patch_len
        if stride is None:
            stride = patch_len
        self.stride = stride
        self.enc_in = enc_in
        self.d_model = d_model
        self.d_ff = d_ff
        self.layers = layers
        self.n_heads = n_heads
        self.position_embedding = position_embedding
        # 计算padding，要求将seq_len整除patch_len
        padding = (configs.seq_len + patch_len - 1) // patch_len * patch_len - configs.seq_len
        # padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout, False)

        # encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),# 这里true是为了让mask生效而不是casual
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.mask_token = nn.Parameter(torch.randn(d_model))
        self.variate_mask_token = nn.Parameter(torch.randn(d_model))
        self.cls_mask_token = nn.Parameter(torch.randn(d_model))

        # Prediction Head
        self.proj = nn.Linear(self.d_model, patch_len, bias=True)
        self.proj_dataset_cls = nn.Linear(self.d_model, 600, bias=True)
        self.proj_variate_cls = nn.Linear(self.d_model, 2, bias=True)

        # patch_encoder_checkpoint = '../ltsm/checkpoints/lotsa_uea_ucr_bert_d1024_l8_p16_n512_weight_p50_wo_revin_full-epoch=25.ckpt'
        patch_encoder_checkpoint = configs.ckpt_path
        if (patch_encoder_checkpoint == '../ltsm/checkpoints/uea_ucr_bert_d1024_l8_p16_n512_weight_p50_wo_revin_full.ckpt'
                or patch_encoder_checkpoint == '../ltsm/checkpoints/lotsa_bert_d768_l12_p16_n512_weight_p50_wo_revin_full-epoch=18.ckpt'):
            self.proj_dataset_cls = nn.Linear(self.d_model, 300, bias=True)


        if configs.use_lm_bert:
            from safetensors.torch import load_file
            state_dict = load_file('./bert2/model.safetensors')
            for k in list(state_dict.keys()):
                # 'bert.encoder.layer.'换成'encoder.attn_layers.'
                if 'bert.encoder.layer.' in k:
                    state_dict[k.replace('bert.encoder.layer.', 'encoder.attn_layers.')] = state_dict.pop(k)
                    k = k.replace('bert.encoder.layer.', 'encoder.attn_layers.')
                # 'attention.output.dense'换成'attention.out_projection'
                if 'attention.output.dense' in k:
                    state_dict[k.replace('attention.output.dense', 'attention.out_projection')] = state_dict.pop(k)
                    k = k.replace('attention.output.dense', 'attention.out_projection')
                # 'attention.self.key' 换成 'attention.key_projection'
                if 'attention.self.key' in k:
                    state_dict[k.replace('attention.self.key', 'attention.key_projection')] = state_dict.pop(k)
                    k = k.replace('attention.self.key', 'attention.key_projection')
                # 'attention.self.query' 换成 'attention.query_projection'
                if 'attention.self.query' in k:
                    state_dict[k.replace('attention.self.query', 'attention.query_projection')] = state_dict.pop(k)
                    k = k.replace('attention.self.query', 'attention.query_projection')
                # 'attention.self.value' 换成 'attention.value_projection'
                if 'attention.self.value' in k:
                    state_dict[k.replace('attention.self.value', 'attention.value_projection')] = state_dict.pop(k)
                    k = k.replace('attention.self.value', 'attention.value_projection')
                # 'attention.output.LayerNorm.gamma' 换成 'norm1.weight'
                if 'attention.output.LayerNorm.gamma' in k:
                    state_dict[k.replace('attention.output.LayerNorm.gamma', 'norm1.weight')] = state_dict.pop(k)
                    k = k.replace('attention.output.LayerNorm.gamma', 'norm1.weight')
                # 'attention.output.LayerNorm.beta' 换成 'norm1.bias'
                if 'attention.output.LayerNorm.beta' in k:
                    state_dict[k.replace('attention.output.LayerNorm.beta', 'norm1.bias')] = state_dict.pop(k)
                    k = k.replace('attention.output.LayerNorm.beta', 'norm1.bias')
                # 'output.LayerNorm.gamma' 换成 'norm2.weight'
                if 'output.LayerNorm.gamma' in k:
                    state_dict[k.replace('output.LayerNorm.gamma', 'norm2.weight')] = state_dict.pop(k)
                    k = k.replace('output.LayerNorm.gamma', 'norm2.weight')
                # 'output.LayerNorm.beta' 换成 'norm2.bias'
                if 'output.LayerNorm.beta' in k:
                    state_dict[k.replace('output.LayerNorm.beta', 'norm2.bias')] = state_dict.pop(k)
                    k = k.replace('output.LayerNorm.beta', 'norm2.bias')
                # 'intermediate.dense' 换成 'conv1'
                if 'intermediate.dense' in k:
                    state_dict[k.replace('intermediate.dense', 'conv1')] = state_dict.pop(k)
                    # intermediate.dense.weight最后挤出一维
                    k = k.replace('intermediate.dense', 'conv1')
                    if 'weight' in k:
                        state_dict[k] = state_dict[k].unsqueeze(-1)

                # 'output.dense' 换成 'conv2'
                if 'output.dense' in k:
                    state_dict[k.replace('output.dense', 'conv2')] = state_dict.pop(k)
                    # output.dense.weight最后挤出一维
                    k = k.replace('output.dense', 'conv2')
                    if 'weight' in k:
                        state_dict[k] = state_dict[k].unsqueeze(-1)
            self.load_state_dict(state_dict, strict=False)
        elif configs.use_vision_bert:
            from safetensors.torch import load_file
            state_dict = load_file('./mae/model.safetensors')
            for k in list(state_dict.keys()):
                # 扔掉含有decoder的参数
                if 'decoder' in k:
                    state_dict.pop(k)
                    continue

                # 'vit.encoder.layer.'换成'encoder.attn_layers.'
                if 'vit.encoder.layer.' in k:
                    state_dict[k.replace('vit.encoder.layer.', 'encoder.attn_layers.')] = state_dict.pop(k)
                    k = k.replace('vit.encoder.layer.', 'encoder.attn_layers.')
                # 'layernorm_after.'norm2.'
                if 'layernorm_after.' in k:
                    state_dict[k.replace('layernorm_after.', 'norm2.')] = state_dict.pop(k)
                    k = k.replace('layernorm_after.', 'norm2.')
                # 'layernorm_before.'norm1.'
                if 'layernorm_before.' in k:
                    state_dict[k.replace('layernorm_before.', 'norm1.')] = state_dict.pop(k)
                    k = k.replace('layernorm_before.', 'norm1.')

                # attention.attention.query. 换成 attention.query_projection
                if 'attention.attention.query.' in k:
                    state_dict[k.replace('attention.attention.query.', 'attention.query_projection.')] = state_dict.pop(k)
                    k = k.replace('attention.attention.query.', 'attention.query_projection.')
                # attention.attention.key. 换成 attention.key_projection
                if 'attention.attention.key.' in k:
                    state_dict[k.replace('attention.attention.key.', 'attention.key_projection.')] = state_dict.pop(k)
                    k = k.replace('attention.attention.key.', 'attention.key_projection.')
                # attention.attention.value. 换成 attention.value_projection
                if 'attention.attention.value.' in k:
                    state_dict[k.replace('attention.attention.value.', 'attention.value_projection.')] = state_dict.pop(k)
                    k = k.replace('attention.attention.value.', 'attention.value_projection.')
                # attention.output.dense 换成 attention.out_projection
                if 'attention.output.dense' in k:
                    state_dict[k.replace('attention.output.dense', 'attention.out_projection')] = state_dict.pop(k)
                    k = k.replace('attention.output.dense', 'attention.out_projection')
                # 'intermediate.dense' 换成 'conv1'
                if 'intermediate.dense' in k:
                    state_dict[k.replace('intermediate.dense', 'conv1')] = state_dict.pop(k)
                    # intermediate.dense.weight最后挤出一维
                    k = k.replace('intermediate.dense', 'conv1')
                    if 'weight' in k:
                        state_dict[k] = state_dict[k].unsqueeze(-1)

                # 'output.dense' 换成 'conv2'
                if 'output.dense' in k:
                    state_dict[k.replace('output.dense', 'conv2')] = state_dict.pop(k)
                    # output.dense.weight最后挤出一维
                    k = k.replace('output.dense', 'conv2')
                    if 'weight' in k:
                        state_dict[k] = state_dict[k].unsqueeze(-1)
            self.load_state_dict(state_dict, strict=False)
        else:
            if not 'random' in patch_encoder_checkpoint:
                if os.path.exists(patch_encoder_checkpoint):
                    sd = torch.load(patch_encoder_checkpoint, map_location="cpu")["state_dict"]
                    # 去掉sd中的前14个字符
                    keys = list(sd.keys())
                    for k in keys:
                        if 'mask_token' == k:
                            # 在前面加上self.
                            sd[f'bert.{k}'] = sd.pop(k)
                    sd = {k[5:]: v for k, v in sd.items()}
                    self.load_state_dict(sd, strict=True)
                else:
                    raise FileNotFoundError(f"File {patch_encoder_checkpoint} not found.")
            else:
                print("Randomly initialized model")

        if not configs.use_finetune:
            for param in self.parameters():
                param.requires_grad = False

        # proj 反向传播
        for param in self.proj.parameters():
            param.requires_grad = True

        self.use_channel_independent = configs.use_channel_independent

        self.pos_embed = self.get_sinusoid_encoding_table(10000, self.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = nn.Linear(self.d_model * ((configs.seq_len + padding) // patch_len + 1), configs.pred_len)
        elif self.task_name == 'classification':
            # self.head = nn.Linear(self.d_model * 31, configs.num_class)
            token_num:int
            if not configs.not_use_dataset_token and not configs.not_use_variate_token:
                if self.use_channel_independent:
                    token_num = ((configs.seq_len + padding) // patch_len + 2) * configs.enc_in
                else:
                    token_num = ((configs.seq_len + padding) // patch_len + 1) * configs.enc_in + 1
            elif not configs.not_use_dataset_token and configs.not_use_variate_token:
                token_num = ((configs.seq_len + padding) // patch_len) * configs.enc_in + 2
            elif configs.not_use_dataset_token and not configs.not_use_variate_token:
                token_num = ((configs.seq_len + padding) // patch_len + 1) * configs.enc_in
            else:
                token_num = ((configs.seq_len + padding) // patch_len) * configs.enc_in

            self.not_use_dataset_token = configs.not_use_dataset_token
            self.not_use_variate_token = configs.not_use_variate_token

            self.head = nn.Linear(self.d_model * token_num, configs.num_class)
            # self.head = nn.Linear(self.d_model, configs.num_class)


    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze(self):
        # 将PatchEmbedding和Encoder的参数冻结
        for param in self.patch_embedding.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False


    def get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        _, N, D = x.shape
        # 前后添加self.cls_mask_token和self.variate_mask_token
        variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
        x = torch.cat([x, variate_mask_token], dim=1) # [B * M, N + 1, D]
        x = x.reshape(B, -1, D) # [B, M*(N + 1), D]


        cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_mask_token, x], dim=1) # [B, M*(N + 1) +1, D]

        _, N_, D = x.shape
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B, N_, -1)
        enc_out, _ = self.encoder(enc_out) # [B, M*(N + 1) +1, D]

        # 取出cls_mask_token输入self.head
        dec_out = self.head(enc_out[:, 1:, :].reshape(B * M, -1)) # [B * M, Pred]
        dec_out = dec_out.view(B, M, -1).transpose(1, 2) # [B, Pred, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):

        if self.use_channel_independent:
            B_, L, M_ = x_enc.shape
            x_enc = x_enc.transpose(1, 2).reshape(-1, L, 1) # [B*M, L, 1]
            mask = mask.transpose(1, 2).reshape(-1, L, 1) # [B*M, L, 1]

        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev


        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        _, N, D = x.shape
        mask = mask.transpose(1, 2).reshape(B * M, N, -1)[:, :, 0].bool()
        x = torch.where(~mask.unsqueeze(-1), self.mask_token, x) # [B * M, N, D]
        variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
        x = torch.cat([x, variate_mask_token], dim=1) # [B * M, N + 1, D]
        x = x.reshape(B, -1, D) # [B, M*(N + 1), D]
        cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_mask_token, x], dim=1) # [B, M*(N + 1) +1, D]

        _, N_, D = x.shape
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B , N_, -1)
        enc_out, _ = self.encoder(enc_out) # [B, M*(N + 1) +1, D]
        # 取出cls_mask_token输入self.head
        dec_out = self.proj(enc_out)# [B, M*(N + 1) +1, P]

        # 将mask_token部分去掉
        dec_out = dec_out[:, 1:, :] # [B, M*(N + 1), P]
        dec_out = dec_out.reshape(B * M, -1, self.patch_len)[:, :N, :] # [B * M, N, P]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2) # [B, L, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev + means

        if self.use_channel_independent:
            dec_out = dec_out.reshape(B_, M_, -1).transpose(1, 2) # [B_, L, M_]

        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):

        if self.use_channel_independent:
            B_, L, M_ = x_enc.shape
            x_enc = x_enc.transpose(1, 2).reshape(-1, L, 1) # [B*M, L, 1]

        B, L, M = x_enc.shape
        # x_enc = x_enc[:, :-8, :]

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        x, n_vars = self.patch_embedding(x_enc) # [B * M, N, D]
        _, N, D = x.shape
        # 前后添加self.cls_mask_token和self.variate_mask_token
        if not self.not_use_variate_token:
            variate_mask_token = self.variate_mask_token.unsqueeze(0).unsqueeze(0).expand(B * M, 1, -1)
            x = torch.cat([x, variate_mask_token], dim=1) # [B * M, N + 1, D]
        x = x.reshape(B, -1, D) # [B, M*(N + 1), D]

        if not self.not_use_dataset_token:
            cls_mask_token = self.cls_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            x = torch.cat([cls_mask_token, x], dim=1)

        _, N_, D = x.shape
        enc_out = x + self.pos_embed.to(x.device)[:, :N_, :].expand(B, N_, -1)
        enc_out, _ = self.encoder(enc_out) # [B, N_, D]
        # 取出cls_mask_token输入self.head
        if self.use_channel_independent:
            dec_out = self.head(enc_out.reshape(B_, -1))
        else:
            dec_out = self.head(enc_out.reshape(B, -1)) # [B, 1, P]

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'mask_reconstruction':
            dec_out = self.mask_reconstruction(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        return None
