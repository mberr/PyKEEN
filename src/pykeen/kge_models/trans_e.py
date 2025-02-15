# -*- coding: utf-8 -*-

"""Implementation of the TransE model."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.autograd
from torch import nn

from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, SCORING_FUNCTION_NORM, TRANS_E_NAME
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = [
    'TransE',
]

log = logging.getLogger(__name__)


class TransE(BaseModule):
    """An implementation of TransE [borders2013]_.

     This model considers a relation as a translation from the head to the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

       - Alternative implementation in OpenKE: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/models/TransE.py
    """

    model_name = TRANS_E_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [SCORING_FUNCTION_NORM, NORM_FOR_NORMALIZATION_OF_ENTITIES]

    def __init__(self,
                 margin_loss: float,
                 embedding_dim: int,
                 scoring_function: Optional[int] = 1,
                 normalization_of_entities: Optional[int] = 2,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu',
                 **kwargs
                 ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)

        self.l_p_norm_entities = normalization_of_entities
        self.scoring_fct_norm = scoring_function

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self._initialize()

    def predict(self, triples):
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print('The model has not been fitted yet. Predictions are based on randomly initialized embeddings.')
            self._init_embeddings()

        triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        positive_scores = self._score_triples(batch_positives)
        negative_scores = self._score_triples(batch_negatives)
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings):
        """Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: emebddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        # Add the vector element wise
        sum_res = head_embeddings + relation_embeddings - tail_embeddings
        distances = torch.norm(sum_res, dim=1, p=self.scoring_fct_norm).view(size=(-1,))
        return distances

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
