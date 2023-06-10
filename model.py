import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from typing import Any, Dict

from core import VisionTransformer
from utils import visualize_prediction


def compute_accuracy(logits, targets):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy


def tensor_to_numpy(tensor):
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    return tensor.numpy()


class VitClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, labels_map=None, *args, **kwargs):
        super().__init__()
        self._core = VisionTransformer(num_classes=num_classes)
        self._criterion = nn.CrossEntropyLoss()
        self._labels_map = labels_map

    @staticmethod
    def preprocess(images_batch, resize=(72, 72)):
        assert len(images_batch.shape) == 4
        if isinstance(images_batch, np.ndarray):
            images_batch = torch.from_numpy(images_batch)
        assert torch.is_tensor(images_batch)
        if images_batch.shape[1] != 3 and images_batch.shape[3] == 3:
            images_batch = torch.permute(images_batch, (0, 3, 1, 2))
        images_batch = images_batch.float() / 255.
        images_batch = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(images_batch)
        if resize is not None:
            images_batch = torchvision.transforms.Resize(resize)(images_batch)
        return images_batch

    def forward(self, images_batch, return_scores=False):
        images_batch = VitClassifier.preprocess(images_batch)
        attn_scores = None
        if return_scores:
            logits, attn_scores = self._core(images_batch, return_scores=True)
        else:
            logits = self._core(images_batch)

        if not self.training:
            scores = torch.softmax(logits, dim=-1)
            max_scores, max_indices = torch.max(scores, dim=1)
            if return_scores:
                return max_scores, max_indices, attn_scores
            else:
                return max_scores, max_indices
        else:
            if return_scores:
                return logits, attn_scores
            else:
                return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [[optimizer], [scheduler]]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self.forward(images)
        loss = self._criterion(logits, labels)
        acc = compute_accuracy(logits, labels)
        self.log_dict(
            {
                f"loss/train": loss.detach(),
                f"accuracy/train": acc,
            }
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        if batch_idx == 0:
            images, labels = batch
            cls_scores, classes, attn_scores = self.forward(images, return_scores=True)
            accuracy = (classes == labels).float().mean()
            self.log_dict(
                {
                    f"accuracy/val": accuracy,
                }
            )
            self.visualize(
                images=images,
                targets=labels,
                pred_labels=classes,
                pred_scores=cls_scores,
                attn_map=attn_scores,
            )

    def visualize(self, images, targets, pred_labels, pred_scores, attn_map):
        predictions, inputs = [], []
        for image, target, pred_label, score, map in \
                zip(images, targets, pred_labels, pred_scores, attn_map):
            image = tensor_to_numpy(image).astype(np.uint8)
            score_str = '{:.3f}'.format(tensor_to_numpy(score))
            target_label = int(tensor_to_numpy(target))
            pred_label = int(tensor_to_numpy(pred_label))
            if self._labels_map is not None:
                target_label = self._labels_map[target_label]
                pred_label = self._labels_map[pred_label]
            text = f'Target: {target_label}'
            text += f'\nPrediction: {pred_label} Score: {score_str}'
            attn = tensor_to_numpy(map)
            predictions.append(
                visualize_prediction(
                    img=image,
                    text=text,
                    attention=attn,
                )
            )
            inputs.append(
                visualize_prediction(
                    img=image,
                    text=text,
                )
            )

        writer = self.logger.experiment

        predictions_tensors = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in predictions]
        predictions_grid = torchvision.utils.make_grid(predictions_tensors)
        writer.add_image(f'Predictions', predictions_grid, self.global_step)

        inputs_tensors = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in inputs]
        inputs_grid = torchvision.utils.make_grid(inputs_tensors)
        writer.add_image(f'Inputs', inputs_grid, self.global_step)
