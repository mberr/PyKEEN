# -*- coding: utf-8 -*-

"""Implementation of TransD."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.autograd
from torch import nn

from pykeen.constants import RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM, TRANS_D_NAME
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = [
    'TransD'
]


class TransD(BaseModule):
    """An implementation of TransD [ji2015]_.

    This model extends TransR to use fewer parameters.

    .. [ji2015] Ji, G., *et al.* (2015). `Knowledge graph embedding via dynamic mapping matrix
                <http://www.aclweb.org/anthology/P15-1067>`_. ACL.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/master/models/TransD.py
    """

    model_name = TRANS_D_NAME
    margin_ranking_loss_size_average: bool = True
    entity_embedding_max_norm = 1
    hyper_params = BaseModule.hyper_params + [RELATION_EMBEDDING_DIM, SCORING_FUNCTION_NORM]

    def __init__(self,
                 margin_loss: float,
                 embedding_dim: int,
                 relation_embedding_dim,
                 scoring_function: Optional[int] = 1,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu',
                 **kwargs
                 ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)

        self.scoring_fct_norm = scoring_function

        # Embeddings
        self.relation_embedding_dim = relation_embedding_dim

    def _init_embeddings(self):
        super()._init_embeddings()
        # A simple lookup table that stores embeddings of a fixed dictionary and size
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_embedding_dim, max_norm=1)
        self.entity_projections = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_projections = nn.Embedding(self.num_relations, self.relation_embedding_dim)
        # FIXME @mehdi what about initialization?

    def predict(self, triples):
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print('The model has not been fitted yet. Predictions are based on randomly initialized embeddings.')
            self._init_embeddings()

        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, positives, negatives):
        positive_scores = self._score_triples(positives)
        negative_scores = self._score_triples(negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        heads, relations, tails = slice_triples(triples)

        h_embs = self._get_entity_embeddings(heads)
        r_embs = self._get_relation_embeddings(relations)
        t_embs = self._get_entity_embeddings(tails)

        h_proj_vec_embs = self._get_entity_projections(heads)
        r_projs_embs = self._get_relation_projections(relations)
        t_proj_vec_embs = self._get_entity_projections(tails)

        proj_heads = self._project_entities(h_embs, h_proj_vec_embs, r_projs_embs)
        proj_tails = self._project_entities(t_embs, t_proj_vec_embs, r_projs_embs)

        scores = self._compute_scores(h_embs=proj_heads, r_embs=r_embs, t_embs=proj_tails)
        return scores

    def _compute_scores(self, h_embs, r_embs, t_embs):
        # Add the vector element wise
        sum_res = h_embs + r_embs - t_embs
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        distances = torch.mul(distances, distances)
        return distances

    def _project_entities(self, entity_embs, entity_proj_vecs, relation_projections):
        relation_projections = relation_projections.unsqueeze(-1)
        entity_proj_vecs = entity_proj_vecs.unsqueeze(-1).permute([0, 2, 1])
        transfer_matrices = torch.matmul(relation_projections, entity_proj_vecs)
        projected_entity_embs = torch.einsum('nmk,nk->nm', [transfer_matrices, entity_embs])
        return projected_entity_embs

    def _get_entity_projections(self, entities):
        return self.entity_projections(entities).view(-1, self.embedding_dim)

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.relation_embedding_dim)

    def _get_relation_projections(self, relations):
        return self.relation_projections(relations).view(-1, self.relation_embedding_dim)
