
def clip_loss(struc_repr, seq_repr, logit_scale):
        # print(struc_repr.shape, seq_repr.shape)
        bt = struc_repr.shape[0]
        assert bt == seq_repr.shape[0]

        struc_repr = struc_repr.reshape(bt, -1) 
        seq_repr = seq_repr.reshape(bt, -1)
        # struc_repr = torch.mean(struc_repr, dim=1, keepdim=False)
        # seq_repr = torch.mean(seq_repr, dim=1, keepdim=False)

        # normalized features
        struc_repr = struc_repr / struc_repr.norm(dim=-1, keepdim=True)
        seq_repr = seq_repr / seq_repr.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = logit_scale.exp()
        logits_per_struc = logit_scale * struc_repr @ seq_repr.T
        logits_per_seq = logits_per_struc.T

        return logits_per_struc, logits_per_seq