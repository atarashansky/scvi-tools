from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Literal, Optional, Sequence

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.module._vae import VAE
from scvi import REGISTRY_KEYS

logger = logging.getLogger(__name__)

class SparseGATEncoder(nn.Module):
    """Graph Attention Network-based encoder with sparsity constraints."""
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        homology_edges: torch.Tensor,
        homology_scores: torch.Tensor,
        n_layers: int = 1,
        n_hidden: int = 128,
        gat_hidden_dim: int = 32,
        n_heads: int = 4,
        dropout_rate: float = 0.1,
        attention_sparsity_weight: float = 0.1,
        temperature: float = 0.5,
        distribution: str = "normal",
    ):
        super().__init__()
        
        self.homology_edges = homology_edges
        self.edge_weights = nn.Parameter(homology_scores)
        self.attention_sparsity_weight = attention_sparsity_weight
        self.temperature = temperature
        self.distribution = distribution
        
        # Input layer
        self.fc_input = nn.Linear(n_input, gat_hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.attention_weights = nn.ParameterList()
        
        curr_dim = gat_hidden_dim
        for i in range(n_layers):
            out_dim = gat_hidden_dim if i < n_layers - 1 else n_hidden
            gat_layer = GATConv(
                curr_dim,
                out_dim,
                heads=n_heads,
                dropout=dropout_rate,
                edge_dim=1  # Edge weights dimension
            )
            self.gat_layers.append(gat_layer)
            self.attention_weights.append(
                nn.Parameter(torch.ones(homology_edges.size(1), n_heads))
            )
            curr_dim = out_dim * n_heads
        
        # Output layers for mean and var
        self.fc_mean = nn.Linear(curr_dim, n_output)
        self.fc_var = nn.Linear(curr_dim, n_output)
        
    def get_sparsity_loss(self) -> torch.Tensor:
        """Calculate L1 sparsity loss on attention weights."""
        return sum(w.abs().mean() for w in self.attention_weights)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape [n_samples, n_features]
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean and variance of the latent distribution
        """
        # Initial feature transformation
        h = F.relu(self.fc_input(x))
        
        # GAT layers
        edge_index = self.homology_edges
        edge_attr = self.edge_weights.unsqueeze(1)  # [E, 1]
        
        for i, (gat_layer, attention_weight) in enumerate(zip(self.gat_layers, self.attention_weights)):
            # Apply gumbel-softmax to attention weights for sparsity
            sparse_attention = F.gumbel_softmax(
                attention_weight,
                tau=self.temperature,
                hard=True
            )
            
            # Combine edge weights with learned attention
            edge_weights = edge_attr * sparse_attention
            
            # Apply GAT layer
            h = gat_layer(h, edge_index, edge_weights)
            if i < len(self.gat_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.1, training=self.training)
        
        # Get mean and var
        mean = self.fc_mean(h)
        var = torch.exp(self.fc_var(h))
        
        return mean, var

class HomologyPredictor(nn.Module):
    """Predicts homology relationships between genes."""
    
    def __init__(
        self,
        n_latent: int,
        n_genes: int,
        batch_size: int = 512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.projector = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarities efficiently in batches.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representations of shape [n_genes, n_latent]
            
        Returns
        -------
        torch.Tensor
            Similarity matrix of shape [n_genes, n_genes]
        """
        z_proj = self.projector(z)
        n_genes = z_proj.size(0)
        
        # Initialize similarity matrix
        sim_matrix = torch.zeros(
            (n_genes, n_genes),
            device=z_proj.device
        )
        
        # Compute similarities batch-wise
        for i in range(0, n_genes, self.batch_size):
            batch_end = min(i + self.batch_size, n_genes)
            batch_z = z_proj[i:batch_end]
            
            # Compute similarities for this batch
            similarities = torch.matmul(batch_z, z_proj.T)
            sim_matrix[i:batch_end] = similarities
        
        # Apply softmax
        return torch.softmax(sim_matrix, dim=-1)

class CrossSpeciesVAE(VAE):
    """VAE model adapted for cross-species integration."""
    
    def __init__(
        self,
        n_input: int,
        homology_edges: torch.Tensor,
        homology_scores: torch.Tensor,
        species_mapping: torch.Tensor,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        homology_loss_weight: float = 0.1,
        attention_sparsity_weight: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        
        self.homology_edges = homology_edges
        self.homology_loss_weight = homology_loss_weight
        self.attention_sparsity_weight = attention_sparsity_weight
        
        # Replace standard encoder with SparseGATEncoder
        self.z_encoder = SparseGATEncoder(
            n_input=n_input,
            n_output=n_latent,
            homology_edges=homology_edges,
            homology_scores=homology_scores,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            attention_sparsity_weight=attention_sparsity_weight,
            distribution=latent_distribution,
        )
        
        # Add homology predictor
        self.homology_predictor = HomologyPredictor(
            n_latent=n_latent,
            n_genes=n_input
        )
    
    @auto_move_data
    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the loss function for the model."""
        # Get original VAE loss
        vae_loss = super().loss(
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight=kl_weight,
        )
        
        # Get latent representations
        z = inference_outputs["z"]
        
        # Compute homology predictions
        sim_matrix = self.homology_predictor(z)
        
        # Positive pairs from homology edges
        pos_loss = -torch.log(
            sim_matrix[self.homology_edges[0], self.homology_edges[1]] + 1e-8
        ).mean()
        
        # Efficient negative sampling using masking
        neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        neg_mask[self.homology_edges[0], self.homology_edges[1]] = False
        neg_loss = -torch.log(1 - sim_matrix[neg_mask] + 1e-8).mean()
        
        # Get sparsity loss from GAT
        sparsity_loss = self.z_encoder.get_sparsity_loss()
        
        # Combine all losses
        total_loss = (
            vae_loss.loss
            + self.homology_loss_weight * (pos_loss + neg_loss)
            + self.attention_sparsity_weight * sparsity_loss
        )
        
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=vae_loss.reconstruction_loss,
            kl_local=vae_loss.kl_local,
            kl_global=vae_loss.kl_global,
            extra_metrics={
                "homology_pos_loss": pos_loss,
                "homology_neg_loss": neg_loss,
                "attention_sparsity_loss": sparsity_loss,
            },
        )
