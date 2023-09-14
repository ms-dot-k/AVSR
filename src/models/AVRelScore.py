#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Korea Advanced Institute of Science and Technology (Joanna Hong, Minsu Kim)

import torch
from torch import nn

class Scoring_Module(torch.nn.Module):
    def __init__(self, indim=512):
        super().__init__()
        self.score = nn.Sequential(nn.Conv1d(indim, indim*2, 7, padding=3),
                                    nn.BatchNorm1d(indim*2),
                                    nn.ReLU(True),
                                    nn.Conv1d(indim*2, indim*2, 7, padding=3),
                                    nn.BatchNorm1d(indim*2),
                                    nn.ReLU(True),
                                    nn.Conv1d(indim*2, indim, 7, padding=3),
                                    nn.BatchNorm1d(indim),
                                    nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1,2).contiguous()
        x = self.score(x)
        x = x.transpose(1,2).contiguous()
        return x

class Scoring(torch.nn.Module):
    def __init__(self, indim=512):
        super().__init__()
        self.score_vid = Scoring_Module(indim)
        self.score_aud = Scoring_Module(indim)

    def generate_key_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [0] * length[i]
            mask += [1] * (sz - length[i])
            masks += [torch.tensor(mask*2)]
        masks = torch.stack(masks, dim=0).bool()
        return masks

    def forward(self, v, a, v_len, is_residual=True, is_scoring=True):
        if not is_scoring:
            out = torch.cat([v, a], dim=1)
            merged_attention_mask = self.generate_key_mask(v_len, v.size(1)).to(out.device)
            return out, ~merged_attention_mask.unsqueeze(1).contiguous()
        else:
            vid_s = self.score_vid(v)
            if is_residual:
                vid_s = v * vid_s + v
            elif is_residual:
                vid_s = v * vid_s
            else:
                raise NotImplementedError
            aud_s = self.score_aud(a)
            if is_residual:
                aud_s = a * aud_s + a
            elif is_residual:
                aud_s = a * aud_s
            else:
                raise NotImplementedError

            out = torch.cat([vid_s, aud_s], dim=1)
            merged_attention_mask = self.generate_key_mask(v_len, vid_s.size(1)).to(out.device)
            return out, ~merged_attention_mask.unsqueeze(1).contiguous()
