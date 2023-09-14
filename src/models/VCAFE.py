#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Korea Advanced Institute of Science and Technology (Joanna Hong, Minsu Kim)

import torch
import torch.nn as nn
import math

class VCA_Masking(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        self.softmax = nn.Softmax(2)
        self.k = nn.Linear(512, out_dim)
        self.v = nn.Linear(512, out_dim)
        self.q = nn.Linear(512, out_dim)
        self.out_dim = out_dim

        self.sigmoid = nn.Sigmoid()
        self.mask = nn.Sequential(
            nn.Conv1d(out_dim, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 1)
        )
        self.dropout = nn.Dropout(0.3)
        self.fusion = nn.Linear(1024, 512)

    def forward(self, aud, vid, v_len):
        #aud: B,T,512
        #vid: B,S,512
        q = self.q(aud)  # B,T,OD
        k = self.k(vid).transpose(1, 2).contiguous()   # B,OD,S

        att = torch.bmm(q, k) / math.sqrt(self.out_dim)    # B,T,S
        for i in range(att.size(0)):
            att[i, :, v_len[i]:] = float('-inf')
        att = self.softmax(att)  # B,T,S

        v = self.v(vid)  # B,S,OD
        value = torch.bmm(att, v)  # B,T,OD

        mask = self.mask(value.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        for i in range(aud.size(0)):
            mask[i, v_len[i]:, :] = float('-inf')
        mask = self.sigmoid(mask)
        enhance_aud = aud * mask
        enhanced_aud = enhance_aud + aud

        fusion = torch.cat([enhanced_aud, vid], 2)
        fusion = self.fusion(self.dropout(fusion))

        return fusion