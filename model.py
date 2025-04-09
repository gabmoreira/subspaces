import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch.linalg import svdvals, eigvalsh, svd, eigh
from resnet import *
from convnet import ConvNet
from dataset import Batch

from typing import Dict, Union, Tuple, Callable, List

import logging

class Model(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        backbone: str,
        activation: Callable,
        alpha: float,
        beta: float,
        kernel: str,
        device: Union[str, torch.device],
    ) -> None:
        super(Model, self).__init__()

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._alpha = alpha
        self._beta = beta
        self._kernel = kernel
        self._device = device

        self._MAX_SIZE = 1024
        self._THRESHOLD = 0.1

        if backbone == "resnet18":
            self._backbone = resnet18(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone == "resnet34":
            self._backbone = resnet34(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone == "resnet50":
            self._backbone = resnet34(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone == "Resnet18":
            self._backbone = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'resnet18',
                pretrained=True)
            self._backbone.fc = nn.Linear(512, dim_out + int(kernel != "linear"))

        elif backbone == "Resnet50":
            self._backbone = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'resnet18',
                pretrained=True)
            self._backbone.fc = nn.Linear(2048, dim_out + int(kernel != "linear"),)

        elif backbone == "convnet":
            self._backbone = ConvNet(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone in ('dinov2_vits14_reg'):
            self._backbone = torch.hub.load('facebookresearch/dinov2', backbone)

        else:
            raise ValueError(f"Unknown backbone {backbone}")

        self._memory_x: List[Tensor] = []
        self._memory_y: List[Tensor] = []

        self._minterms: Dict[Tuple, Tensor] = {}
        self._minterm_samples: Dict[Tuple, Tensor] = {}

    def forget(self) -> None:
        self._memory_x = []
        self._memory_y = []
        logging.info("Memory reset")

    def remember(self, x: Tensor, data: Batch) -> None:
        self._memory_x.append(x.detach().cpu())
        self._memory_y.append(data.labels.cpu().to(torch.int32))

    def embed(self, data: Batch) -> Tensor:
        if "dinov2" in str(type(self._backbone)):
            out = self._backbone.forward_features(data.images)
            feat = out["x_prenorm"][:,0]
        else:
            feat = self._backbone(data.images)

        if self._kernel == "linear":
            return feat, None
        else:
            x, norm = torch.split(feat, (self._dim_out, 1), -1)
            return x, norm

    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self._kernel == "linear":
            return x1 @ x2.T
        elif self._kernel == "gaussian":
            return torch.exp(-(torch.cdist(x1, x2, p=2)**2))
        else:
            raise NotImplemented(f"Unknown kernel {self._kernel}")

    def forward(self, data: Batch) -> Dict[str, Tensor]:
        x, norm = self.embed(data)
        y = data.labels.float()

        if self._kernel == "linear":
            z = torch.cat((y.T, x.T), dim=0)
            z_svals = svdvals(z)
            x_svals = svdvals(x)
        else:
            kernel = norm.view(1,-1) * self.kernel(x,x) * norm.view(-1,1)
            z_svals = torch.sqrt(F.relu(eigvalsh(y @ y.T + kernel)))
            x_svals = torch.sqrt(F.relu(eigvalsh(kernel)))

        self.remember(x, data)

        loss = z_svals.sum()
        loss -= self._alpha * x_svals.sum()
        loss += self._beta * x_svals.max() ** 2

        output = {"x_norm" : x_svals.sum(),
                  "z_norm" : z_svals.sum(),
                  "loss" : loss}

        return output

    @torch.no_grad()
    def update_minterms(self) -> None:
        if len(self._memory_x) == 0:
            return None

        memory_x = torch.cat(self._memory_x, dim=0).to(self._device)
        memory_y = torch.cat(self._memory_y, dim=0).to(self._device)
        minterms = torch.unique(memory_y, dim=0)

        for minterm in minterms:
            key = tuple(minterm.cpu().to(torch.int32).tolist())

            mask = (memory_y == minterm).all(dim=-1)
            minterm_samples = memory_x[mask]
            minterm_samples = minterm_samples[-self._MAX_SIZE:]
            self._minterm_samples[key] = minterm_samples

            if self._kernel == "linear":
                minterm_samples = F.normalize(minterm_samples, dim=-1, p=2)
                try:
                    U, _, _ = svd(minterm_samples.T)
                except:
                    logging.error(f"svd error minterm {key}")
                else:
                    self._minterms[key] = U[:,:1].T
            else:
                K = self.kernel(minterm_samples, minterm_samples)
                try:
                    lbds, U = eigh(K)
                except:
                    logging.error(f"eigh error minterm {key}")
                else:
                    p = lbds / K.shape[0]
                    idx = torch.cumsum(p, 0) > self._THRESHOLD
                    self._minterms[key] = U[:,idx].T / torch.sqrt(lbds[idx].view(-1,1))

    @torch.no_grad()
    def evaluate(self, data: Batch) -> Tuple[Tensor, Tensor]:    
        if len(self._minterms) == 0:
            return torch.tensor([0]), torch.tensor([0])    
        
        batch_size = data.images.shape[0]

        # Create minterm labels (ficticious labels)
        minterm_labels = torch.full((batch_size,), -1, dtype=torch.int32, device=self._device)
        minterm_evecs = []
        for i, minterm in enumerate(self._minterms.keys()):
            mask = (data.labels == torch.tensor(minterm, device=self._device)).all(dim=-1)
            minterm_labels.masked_fill_(mask, i)
            minterm_evecs.append(self._minterms[minterm])

        logging.info(f"Eval with {len(self._minterms)} minterms")
        logging.info(f"Eval found {minterm_labels[minterm_labels < 0].sum()} unknown samples")
        
        x_query, _ = self.embed(data)

        if self._kernel == "linear":
            minterm_evecs = torch.cat(minterm_evecs)
            x_query = F.normalize(x_query, dim=-1, p=2)
            p = torch.square(x_query @ minterm_evecs.T)
        else:
            p = []
            for minterm, minterm_evec in self._minterms.items():
                # (batch_size, n_minterm_samples)
                kernel_memory_query = self.kernel(x_query, self._minterm_samples[minterm])

                # (batch_size, n_evecs)
                projections = torch.einsum(
                    "bj,kj->bk",
                    kernel_memory_query,
                    minterm_evec,
                )
                
                p.append(torch.square(projections).sum(-1).view(-1,1))
            p = torch.cat(p, dim=-1)
        
        p = p / p.sum(-1, keepdim=True)
        minterm_predictions = torch.argmax(p, dim=-1)

        label_probs = []
        minterms = torch.tensor([k for k in self._minterms.keys()], device=self._device)
        for label_idx in range(minterms.shape[1]):
            mask = minterms[:,label_idx] == 1
            label_probs.append(p[:,mask].sum(-1))
        label_probs = torch.stack(label_probs, dim=-1)
        return minterm_predictions[minterm_labels > -1], minterm_labels[minterm_labels > -1], label_probs


class XentModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        backbone: str,
        activation: Callable,
        alpha: float,
        beta: float,
        kernel: str,
        device: Union[str, torch.device],
    ) -> None:
        super(XentModel, self).__init__()

        self._dim_in = dim_in
        self._dim_out = dim_out
        self._device = device

        self.criterion = torch.nn.BCEWithLogitsLoss()
        if backbone == "resnet18":
            self._backbone = resnet18(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone == "resnet34":
            self._backbone = resnet34(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone == "resnet50":
            self._backbone = resnet34(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone == "Resnet18":
            self._backbone = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'resnet18',
                pretrained=True)
            self._backbone.fc = nn.Linear(512, dim_out + int(kernel != "linear"))

        elif backbone == "Resnet50":
            self._backbone = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'resnet18',
                pretrained=True)
            self._backbone.fc = nn.Linear(2048, dim_out + int(kernel != "linear"),)

        elif backbone == "convnet":
            self._backbone = ConvNet(
                in_dim=dim_in,
                out_dim=dim_out + int(kernel != "linear"),
                activation=activation
            )

        elif backbone in ('dinov2_vits14_reg'):
            self._backbone = torch.hub.load('facebookresearch/dinov2', backbone)

        else:
            raise ValueError(f"Unknown backbone {backbone}")

        self._memory_x: List[Tensor] = []
        self._memory_y: List[Tensor] = []

        self._minterms: Dict[Tuple, Tensor] = {}
        self._minterm_samples: Dict[Tuple, Tensor] = {}

    def forget(self) -> None:
        pass

    def remember(self, x: Tensor, data: Batch) -> None:
        pass

    def embed(self, data: Batch) -> Tensor:
        if "dinov2" in str(type(self._backbone)):
            out = self._backbone.forward_features(data.images)
            feat = out["x_prenorm"][:,0]
        else:
            feat = self._backbone(data.images)

        return feat, None
    
    def kernel(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self._kernel == "linear":
            return x1 @ x2.T
        elif self._kernel == "gaussian":
            return torch.exp(-(torch.cdist(x1, x2, p=2)**2))
        else:
            raise NotImplemented(f"Unknown kernel {self._kernel}")

    def forward(self, data: Batch) -> Dict[str, Tensor]:
        logits, _ = self.embed(data)
        y = data.labels.float()
        loss = self.criterion(logits, y)
        output = {"loss" : loss}
        return output

    @torch.no_grad()
    def update_minterms(self) -> None:
        pass

    @torch.no_grad()
    def evaluate(self, data: Batch) -> Tuple[Tensor, Tensor]:            
        logits, _ = self.embed(data)
        label_probs = F.sigmoid(logits)
        return torch.ones(logits.shape[0]), torch.ones(logits.shape[0]), label_probs