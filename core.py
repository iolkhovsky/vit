import torch
import torch.nn as nn
import torchvision.models as models

from layers import (
    PatchExtractor, PatchEncoder, TransformerEncoder, MLP
)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, embedding_dims=64,
                 encoder_blocks=8, num_classes=10):
        super(VisionTransformer, self).__init__()
        self._patch_ext = PatchExtractor(patch_size)
        self._attn_map_size = [img_size // patch_size] * 2
        patches_n = (img_size // patch_size) ** 2
        self._patch_encoder = PatchEncoder(
            patch_shape=(patches_n, 3, patch_size, patch_size),
            embedding_dims=embedding_dims,
        )
        self._norm = nn.LayerNorm(embedding_dims)
        self._do = nn.Dropout(p=0.5)
        self._encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embedding_dims=embedding_dims,
                    num_heads=4,
                ) for _ in range(encoder_blocks)
            ]
        )
        mlp_features = [
            patches_n * embedding_dims,
            2048,
            1024
        ]
        self._mlp = MLP(
            features=mlp_features,
            do=0.5
        )
        self._classifier = nn.Linear(
            in_features=mlp_features[-1],
            out_features=num_classes,
        )

    def forward(self, x, return_scores=False):
        patches = self._patch_ext(x)
        embeddings = self._patch_encoder(patches)

        for _, layer in enumerate(self._encoders[:-1]):
            embeddings = layer(embeddings)
        if return_scores:
            encoded_seq, scores = self._encoders[-1](embeddings, return_scores=True)
            b = encoded_seq.shape[0]
            scores = torch.mean(scores, dim=1).reshape([b,] + self._attn_map_size)
        else:
            encoded_seq = self._encoders[-1](embeddings)

        norm_seq = self._norm(encoded_seq)
        flatten_seq = torch.reshape(norm_seq, [norm_seq.shape[0], -1])
        flatten_seq = self._do(flatten_seq)

        features = self._mlp(flatten_seq)
        logits = self._classifier(features)
        if return_scores:
            return logits, scores
        else:
            return logits
