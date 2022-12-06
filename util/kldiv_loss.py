import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits):
        # prediction_logits, ref_logits => B,L,D

        prediction_logits = F.log_softmax(prediction_logits/self.T, dim=-1).view(-1, ref_logits.shape[2])
        ref_probs = F.softmax(ref_logits/self.T, dim=-1).view(-1, ref_logits.shape[2])
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T

        return loss