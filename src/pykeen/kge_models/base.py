# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import logging
import timeit
from tqdm import trange
from typing import Any, List, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.optim as optim

from pykeen.constants import (
    EMBEDDING_DIM, GPU, LEARNING_RATE, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE,
)

__all__ = [
    'BaseModule',
    'slice_triples'
]

log = logging.getLogger(__name__)


def get_device(device):
    """Get the Torch device to use."""
    if device == 'gpu':
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device('cpu')
    return torch_device

class BaseModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE]

    def __init__(self, margin_loss, num_entities, num_relations, embedding_dim,
                 random_seed: Optional[int] = None,
                 preferred_device: Optional[str] = 'cpu') -> None:
        super().__init__()

        # Device selection
        self.device = get_device(preferred_device)

        self.random_seed = random_seed

        # Loss
        self.margin_loss = margin_loss
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            size_average=self.margin_ranking_loss_size_average,
        )

        # Entity dimensions
        #: The number of entities in the knowledge graph
        self.num_entities = num_entities
        #: The number of unique relation types in the knowledge graph
        self.num_relations = num_relations
        #: The dimension of the embeddings to generate
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(
            self.num_entities,
            self.embedding_dim,
            norm_type=self.entity_embedding_norm_type,
            max_norm=self.entity_embedding_max_norm,
        )

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _to_cpu(self):
        self.device = torch.device('cpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _to_gpu(self):
        self.device = torch.device('cuda')
        self.to(self.device)
        torch.cuda.empty_cache()

    def fit(
            self,
            all_entities,
            pos_triples,
            learning_rate,
            num_epochs,
            batch_size,
            tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        """"""
        self.to(self.device)

        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)

        if self.model_name == 'CONV_E_NAME':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        log.info('****Run Model On %s****' % str(self.device).upper())

        loss_per_epoch = []
        num_pos_triples = pos_triples.shape[0]
        num_entities = all_entities.shape[0]

        start_training = timeit.default_timer()

        _tqdm_kwargs = dict(desc='Training epoch')
        if tqdm_kwargs:
            _tqdm_kwargs.update(tqdm_kwargs)

        for epoch in trange(num_epochs, **_tqdm_kwargs):
            indices = np.arange(num_pos_triples)
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            num_positives = batch_size // 2
            pos_batches = _split_list_in_batches(input_list=pos_triples, batch_size=num_positives)
            current_epoch_loss = 0.

            for i, pos_batch in enumerate(pos_batches):
                current_batch_size = len(pos_batch)
                batch_subjs = pos_batch[:, 0:1]
                batch_relations = pos_batch[:, 1:2]
                batch_objs = pos_batch[:, 2:3]

                num_subj_corrupt = len(pos_batch) // 2
                num_obj_corrupt = len(pos_batch) - num_subj_corrupt
                pos_batch = torch.tensor(pos_batch, dtype=torch.long, device=self.device)

                corrupted_subj_indices = np.random.choice(np.arange(0, num_entities), size=num_subj_corrupt)
                corrupted_subjects = np.reshape(all_entities[corrupted_subj_indices], newshape=(-1, 1))
                subject_based_corrupted_triples = np.concatenate(
                    [corrupted_subjects, batch_relations[:num_subj_corrupt], batch_objs[:num_subj_corrupt]], axis=1)

                corrupted_obj_indices = np.random.choice(np.arange(0, num_entities), size=num_obj_corrupt)
                corrupted_objects = np.reshape(all_entities[corrupted_obj_indices], newshape=(-1, 1))

                object_based_corrupted_triples = np.concatenate(
                    [batch_subjs[num_subj_corrupt:], batch_relations[num_subj_corrupt:], corrupted_objects], axis=1)

                neg_batch = np.concatenate([subject_based_corrupted_triples, object_based_corrupted_triples], axis=0)

                neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=self.device)

                # Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                optimizer.zero_grad()
                loss = self(pos_batch, neg_batch)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                optimizer.step()

            # log.info("Epoch %s took %s seconds \n" % (str(epoch), str(round(stop - start))))
            # Track epoch loss
            loss_per_epoch.append(current_epoch_loss / len(pos_triples))

        stop_training = timeit.default_timer()
        log.info("Training took %s seconds \n" % (str(round(stop_training - start_training))))

        return loss_per_epoch


def slice_triples(triples):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t

def _split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]