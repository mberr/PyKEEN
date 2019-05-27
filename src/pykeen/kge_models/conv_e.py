# -*- coding: utf-8 -*-

"""Implementation of ConvE."""

from typing import Dict

import torch
import torch.autograd
from torch import nn
from torch.nn import Parameter, functional as F
from torch.nn.init import xavier_normal_
from pykeen.kge_models.base import BaseModule, slice_triples
from typing import Dict, Optional
import torch.optim as optim
from functools import lru_cache

from pykeen.constants import (
    CONV_E_FEATURE_MAP_DROPOUT, CONV_E_HEIGHT, CONV_E_INPUT_CHANNELS, CONV_E_INPUT_DROPOUT, CONV_E_KERNEL_HEIGHT,
    CONV_E_KERNEL_WIDTH, CONV_E_NAME, CONV_E_OUTPUT_CHANNELS, CONV_E_OUTPUT_DROPOUT, CONV_E_WIDTH, EMBEDDING_DIM,
    NUM_ENTITIES, NUM_RELATIONS,
    MARGIN_LOSS, LEARNING_RATE, PREFERRED_DEVICE, GPU)

__all__ = ['ConvE']


class ConvE(BaseModule):
    """An implementation of ConvE [dettmers2017]_.

    .. [dettmers2017] Dettmers, T., *et al.* (2017) `Convolutional 2d knowledge graph embeddings
                      <https://arxiv.org/pdf/1707.01476.pdf>`_. arXiv preprint arXiv:1707.01476.

    .. seealso:: https://github.com/TimDettmers/ConvE/blob/master/model.py
    """

    model_name = CONV_E_NAME
    hyper_params = [EMBEDDING_DIM, CONV_E_INPUT_CHANNELS, CONV_E_OUTPUT_CHANNELS, CONV_E_HEIGHT, CONV_E_WIDTH,
                    CONV_E_KERNEL_HEIGHT, CONV_E_KERNEL_WIDTH, CONV_E_INPUT_DROPOUT, CONV_E_FEATURE_MAP_DROPOUT,
                    CONV_E_OUTPUT_DROPOUT, MARGIN_LOSS, LEARNING_RATE]

    def __init__(self,
                 margin_loss: float,
                 embedding_dim: int,
                 ConvE_input_channels, ConvE_output_channels, ConvE_height, ConvE_width, ConvE_kernel_height,
                 ConvE_kernel_width, conv_e_input_dropout, conv_e_output_dropout, conv_e_feature_map_dropout,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu',
                 **kwargs
                 ) -> None:
        super().__init__(margin_loss, embedding_dim, random_seed, preferred_device)

        self.ConvE_height = ConvE_height
        self.ConvE_width = ConvE_width

        assert self.ConvE_height * self.ConvE_width == self.embedding_dim

        self.inp_drop = torch.nn.Dropout(conv_e_input_dropout)
        self.hidden_drop = torch.nn.Dropout(conv_e_output_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(conv_e_feature_map_dropout)
        # self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=ConvE_input_channels,
            out_channels=ConvE_output_channels,
            kernel_size=(ConvE_kernel_height, ConvE_kernel_width),
            stride=1,
            padding=0,
            bias=True,
        )

        # num_features – C from an expected input of size (N,C,L)
        self.bn0 = torch.nn.BatchNorm2d(ConvE_input_channels)
        # num_features – C from an expected input of size (N,C,H,W)
        self.bn1 = torch.nn.BatchNorm2d(ConvE_output_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)
        num_in_features = ConvE_output_channels * \
                          (2 * self.ConvE_height - ConvE_kernel_height + 1) * \
                          (self.ConvE_width - ConvE_kernel_width + 1)
        self.fc = torch.nn.Linear(num_in_features, self.embedding_dim)

        # Default optimizer for ConvE
        self.default_optimizer = optim.Adam

    def _init_embeddings(self):
        super()._init_embeddings()
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(self.num_entities)))
        xavier_normal_(self.entity_embeddings.weight.data)
        xavier_normal_(self.relation_embeddings.weight.data)


    def forward_conv(self, subject, relation):
        batch_size = subject.shape[0]
        subject_batch_embedded = self.entity_embeddings(subject).view(-1, 1, self.ConvE_height, self.ConvE_width)
        relation_batch_embedded = self.relation_embeddings(relation).view(-1, 1,
                                                                                self.ConvE_height, self.ConvE_width)
        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = torch.cat([subject_batch_embedded, relation_batch_embedded], 2)

        # batch_size, num_input_channels, 2*height, width
        stacked_inputs = self.bn0(stacked_inputs)

        # batch_size, num_input_channels, 2*height, width
        x = self.inp_drop(stacked_inputs)
        # (N,C_out,H_out,W_out)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # batch_size, num_output_channels * (2 * height - ConvE_kernel_height + 1) * (width - ConvE_kernel_width + 1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)

        return x

    def predict_right(self, triple):
        """
        Score all possible objects except the object of the true triple
        :param triple:
        :return:
        """
        subject = triple[0:1]
        relation = triple[1:2]
        object = triple[2:3]

        all_entities = torch.arange(self.num_entities, device=self.device)

        object_batch = all_entities[all_entities!=object]

        candidate_object_embeddings = self.entity_embeddings(object_batch)

        x = self.forward_conv(subject, relation)
        x = torch.mm(x, candidate_object_embeddings.transpose(1, 0)).flatten()

        scores = torch.sigmoid(x)

        # Class 0 represents false fact and class 1 represents true fact

        return scores

    def predict_left(self, triple):
        """
        Score all possible subjects except the subject of the true triple
        :param triple:
        :return:
        """
        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        subject = triple[0:1]
        relation = triple[1:2]
        object = triple[2:3]

        all_entities = torch.arange(self.num_entities, device=self.device)

        subject_batch = all_entities[all_entities!=subject]
        relation_batch = torch.repeat_interleave(relation, self.num_entities)

        candidate_object_embeddings = self.entity_embeddings(object)

        x = self.forward_conv(subject_batch, relation_batch)
        x = torch.mm(x, candidate_object_embeddings.transpose(1, 0)).flatten()

        scores = torch.sigmoid(x)

        # Class 0 represents false fact and class 1 represents true fact

        return scores

    def predict_for_ranking(self, subject_batch, relation_batch, object_batch):
        """
        Score all possible subjects except the subject of the true triple
        :param triple:
        :return:
        """

        candidate_object_embeddings = self.entity_embeddings(object_batch)

        x = self.forward_conv(subject_batch, relation_batch)
        x = torch.mm(x, candidate_object_embeddings.transpose(1, 0)).flatten()

        scores = torch.sigmoid(x)

        # Class 0 represents false fact and class 1 represents true fact

        return scores


    def predict(self, triples):
        # Check if the model has been fitted yet.
        if self.entity_embeddings is None:
            print('The model has not been fitted yet. Predictions are based on randomly initialized embeddings.')
            self._init_embeddings()

        # triples = torch.tensor(triples, dtype=torch.long, device=self.device)
        batch_size = triples.shape[0]
        subject_batch = triples[:, 0:1]
        relation_batch = triples[:, 1:2]
        object_batch = triples[:, 2:3].view(-1)

        candidate_object_embeddings = self.entity_embeddings(object_batch)

        x = self.forward_conv(subject_batch, relation_batch)
        x = torch.sum(torch.mul(x.flatten(), candidate_object_embeddings.flatten()).reshape(batch_size, -1), dim=1)

        scores = torch.sigmoid(x)

        # Class 0 represents false fact and class 1 represents true fact
        return scores

    def forward(self, pos_batch, labels):
        batch = pos_batch

        batch_size = batch.shape[0]

        heads = batch[:, 0:1]
        relations = batch[:, 1:2]
        # tails = batch[:, 2:3]

        # TODO: Make smart unpacking of all values for labels
        # labels_full = torch.zeros((batch_size, self.num_entities), device=self.device)
        # for i in range(batch_size):
        #     labels_full[i, labels[i]] = 1

        x = self.forward_conv(heads, relations)

        x = torch.mm(x, self.entity_embeddings.weight.transpose(1,0))

        predictions = x

        loss = self.loss(predictions, labels)
        return loss