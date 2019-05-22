# -*- coding: utf-8 -*-

"""Implementation of ERMLP."""

from typing import Dict, Optional

import torch
import torch.autograd
from torch import nn
from torch.nn.init import xavier_normal_

from pykeen.constants import ERMLP_NAME
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = ['ERMLP']


class ERMLP(BaseModule):
    """An implementation of ERMLP [dong2014]_.

    This model uses a neural network-based approach.

    .. [dong2014] Dong, X., *et al.* (2014) `Knowledge vault: A web-scale approach to probabilistic knowledge fusion
                  <https://dl.acm.org/citation.cfm?id=2623623>`_. ACM.
    """

    model_name = ERMLP_NAME
    margin_ranking_loss_size_average: bool = True

    def __init__(self,
                 margin_loss: float,
                 embedding_dim: int,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu',
                 **kwargs
                 ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)

        """The mulit layer perceptron consisting of an input layer with 3 * self.embedding_dim neurons, a  hidden layer
           with self.embedding_dim neurons and output layer with one neuron.
           The input is represented by the concatenation embeddings of the heads, relations and tail embeddings.
        """
        #: Embeddings for relations in the knowledge graph
        self.relation_embeddings = None

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
        )

        self.mlp.apply(self._init_weights)

        self.criterion = nn.BCELoss()

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.01)

    def _init_embeddings(self):
        super()._init_embeddings()
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

    def predict(self, triples):
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print('The model has not been fitted yet. Predictions are based on randomly initialized embeddings.')
            self._init_embeddings()

        triples = torch.tensor(triples, dtype=torch.long, device=self.device)

        scores = self._score_triples(triples)
        predictions = torch.sigmoid(scores)

        return predictions.detach().cpu().numpy()

    def forward(self, pos_batch, neg_batch):
        batch = torch.cat((pos_batch, neg_batch), dim=0)
        positive_labels = torch.ones(pos_batch.shape[0], dtype=torch.float, device=self.device)
        negative_labels = torch.zeros(neg_batch.shape[0], dtype=torch.float, device=self.device)
        labels = torch.cat([positive_labels, negative_labels], dim=0)

        perm = torch.randperm(labels.shape[0])

        batch = batch[perm]
        labels = labels[perm]

        scores = self._score_triples(batch)
        predictions = torch.sigmoid(scores)

        loss = self.criterion(predictions, labels)
        return loss

    # def forward(self, positives, negatives):
    #
    #     positive_scores = self._score_triples(positives)
    #     negative_scores = self._score_triples(negatives)
    #     loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
    #     return loss

    def _compute_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        y = torch.FloatTensor([-1])
        y = y.expand(positive_scores.shape[0]).to(self.device)
        loss = self.criterion(positive_scores, negative_scores, y)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings = self._get_triple_embeddings(triples)
        scores = self._compute_scores(head_embeddings, relation_embeddings, tail_embeddings)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings):
        # batch_size, num_input_channels, width, height
        heads_embs = head_embeddings.view(-1, self.embedding_dim)
        relation_embs = relation_embeddings.view(-1, self.embedding_dim)
        tails_embs = tail_embeddings.view(-1, self.embedding_dim)

        # batch_size, num_input_channels, 2*height, width
        concat_inputs = torch.cat([heads_embs, relation_embs], 1)

        mlp_output = self.mlp(concat_inputs)

        scores = torch.sum(torch.mm(mlp_output, tails_embs.transpose(1, 0)), dim=1)
        return scores

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)
