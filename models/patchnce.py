from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # CRITICAL: L2 normalize features to prevent extreme dot products at high resolution
        # This makes similarity scores bounded to [-1, 1]
        feat_q = torch.nn.functional.normalize(feat_q, p=2, dim=1)
        feat_k = torch.nn.functional.normalize(feat_k, p=2, dim=1)

        # Check for NaN after normalization
        if torch.isnan(feat_q).any() or torch.isnan(feat_k).any():
            print("WARNING: NaN detected in NCE features after normalization")
            # Return zero loss if features are invalid
            return torch.zeros(num_patches, device=feat_q.device)

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # CRITICAL: Clamp logits to prevent exp() overflow in cross-entropy
        # At high resolution, even normalized features can produce extreme logits when divided by small temperature
        out = torch.clamp(out, min=-20.0, max=20.0)
        
        # Check for NaN/Inf before loss computation
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("WARNING: NaN/Inf detected in NCE logits")
            return torch.zeros(num_patches, device=feat_q.device)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        
        # Final safety check
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("WARNING: NaN/Inf detected in final NCE loss")
            return torch.zeros(num_patches, device=feat_q.device)

        return loss
