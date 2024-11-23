from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, CategoricalObsField, NumericalObsField
from scvi.model.base import BaseModelClass, VAEMixin, UnsupervisedTrainingMixin
from scvi.module import CrossSpeciesVAE
from scvi.utils import setup_anndata_dsp

logger = logging.getLogger(__name__)

class CrossSpeciesSCVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Single-cell Variational Inference model adapted for cross-species integration.
    
    Parameters
    ----------
    adata
        AnnData object containing gene expression data
    homology_edges
        Edge indices for homologous genes (2 x num_edges)
    homology_scores
        Initial scores for homologous relationships (num_edges,)
    species_key
        Key in adata.obs containing species information
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    homology_loss_weight
        Weight for the homology prediction loss
    attention_sparsity_weight
        Weight for the attention sparsity loss
    **model_kwargs
        Additional keyword arguments for :class:`~scvi.module.CrossSpeciesVAE`
    """
    
    def __init__(
        self,
        adata: AnnData,
        homology_edges: np.ndarray,
        homology_scores: np.ndarray,
        species_key: str,
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
        super().__init__(adata)
        
        # Convert numpy arrays to torch tensors
        homology_edges = torch.from_numpy(homology_edges).long()
        homology_scores = torch.from_numpy(homology_scores).float()
        
        # Get species information
        species_labels = self.adata.obs[species_key].cat.codes.values
        species_mapping = torch.from_numpy(species_labels).long()
        
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        
        # Initialize the VAE module
        self.module = CrossSpeciesVAE(
            n_input=self.summary_stats.n_vars,
            homology_edges=homology_edges,
            homology_scores=homology_scores,
            species_mapping=species_mapping,
            n_batch=self.summary_stats.n_batch,
            n_labels=self.summary_stats.n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            homology_loss_weight=homology_loss_weight,
            attention_sparsity_weight=attention_sparsity_weight,
            **model_kwargs,
        )
        self._model_summary_string = "CrossSpeciesSCVI"
        self.init_params_ = self._get_init_params(locals())
    
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[Sequence[str]] = None,
        continuous_covariate_keys: Optional[Sequence[str]] = None,
        species_key: Optional[str] = None,
    ):
        """
        Sets up AnnData object for cross-species integration.
        
        Parameters
        ----------
        adata
            AnnData object containing gene expression data
        layer
            If not None, use this as the key in adata.layers for raw count data
        batch_key
            Key in adata.obs for batch information
        labels_key
            Key in adata.obs for label information
        size_factor_key
            Key in adata.obs for size factor information
        categorical_covariate_keys
            Keys in adata.obs for categorical covariates
        continuous_covariate_keys
            Keys in adata.obs for continuous covariates
        species_key
            Key in adata.obs for species information
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(REGISTRY_KEYS.SPECIES_KEY, species_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **setup_method_args)
        cls.register_manager(adata_manager)
