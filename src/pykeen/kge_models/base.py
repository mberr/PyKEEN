# -*- coding: utf-8 -*-

"""Utilities for getting and initializing KGE models."""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import logging
import timeit
from tqdm import trange
from typing import Any, List, Mapping, Optional, Tuple
import random

import torch
from torch import nn
import torch.optim as optim

from pykeen.constants import (
    EMBEDDING_DIM, GPU, LEARNING_RATE, MARGIN_LOSS, NUM_ENTITIES, NUM_RELATIONS, PREFERRED_DEVICE,
)

__all__ = [
    'BaseModule',
    'slice_triples',
]

log = logging.getLogger(__name__)

class BaseModule(nn.Module):
    """A base class for all of the models."""

    margin_ranking_loss_size_average: bool = ...
    entity_embedding_max_norm: Optional[int] = None
    entity_embedding_norm_type: int = 2
    hyper_params = [EMBEDDING_DIM, MARGIN_LOSS, LEARNING_RATE]

    def __init__(self,
                 margin_loss: float,
                 num_entities: int,
                 num_relations: int,
                 embedding_dim: int,
                 random_seed: Optional[int] = None,
                 preferred_device: str = 'cpu',
                 ) -> None:
        super().__init__()

        # Device selection
        self._get_device(preferred_device)

        self.random_seed = random_seed

        # Random seeds have to set before the embeddings are initialized
        if self.random_seed is not None:
            np.random.seed(seed=self.random_seed)
            torch.manual_seed(seed=self.random_seed)
            random.seed(self.random_seed)

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

        # Default optimizer for all classes
        self.default_optimizer = optim.SGD

    def __init_subclass__(cls, **kwargs):  # noqa: D105
        if not getattr(cls, 'model_name', None):
            raise TypeError('missing model_name class attribute')

    def _get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities).view(-1, self.embedding_dim)

    def _get_device(self,
                    device: str = 'cpu',
                    ) -> None:
        """Get the Torch device to use."""
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                print('No cuda devices were available. The model runs on CPU')
        else:
            self.device = torch.device('cpu')

    def _to_cpu(self):
        """Transfer the entire model to CPU"""
        self._get_device('cpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def _to_gpu(self):
        """Transfer the entire model to GPU"""
        self._get_device('gpu')
        self.to(self.device)
        torch.cuda.empty_cache()

    def fit(
            self,
            all_entities: np.ndarray,
            pos_triples: np.ndarray,
            learning_rate: float,
            num_epochs: int,
            batch_size: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        """
        Trains the kge model with the given parameters
        :param all_entities: List of all entities
        :param pos_triples: Positive triples to train on
        :param learning_rate: Learning rate for the optimizer
        :param num_epochs: Number of epochs to train
        :param batch_size: Batch size for training
        :param optimizer: Pytorch optimizer class to use for training
        :param tqdm_kwargs: Keyword arguments that should be used for the tdqm.trange class
        :return: loss_per_epoch: The loss of each epoch during training
        """
        self.to(self.device)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        if optimizer is None:
            # Initialize the standard optimizer with the correct parameters
            self.optimizer = self.default_optimizer(self.parameters(), lr=self.learning_rate)
        else:
            # Initialize the optimizer given as attribute
            self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)

        log.info(f'****Run Model On {str(self.device).upper()}****')

        loss_per_epoch = []
        num_pos_triples = pos_triples.shape[0]
        num_entities = all_entities.shape[0]

        start_training = timeit.default_timer()

        _tqdm_kwargs = dict(desc='Training epoch')
        if tqdm_kwargs:
            _tqdm_kwargs.update(tqdm_kwargs)

        for epoch in trange(self.num_epochs, **_tqdm_kwargs):
            indices = np.arange(num_pos_triples)
            np.random.shuffle(indices)
            pos_triples = pos_triples[indices]
            num_positives = self.batch_size // 2
            pos_batches = _split_list_in_batches(input_list=pos_triples, batch_size=num_positives)
            current_epoch_loss = 0.

            for i, pos_batch in enumerate(pos_batches):
                # TODO: Implement helper functions for different negative sampling approaches
                current_batch_size = len(pos_batch)

                batch_subjs, batch_relations, batch_objs = slice_triples(pos_batch)

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
                self.optimizer.zero_grad()
                loss = self(pos_batch, neg_batch)
                current_epoch_loss += (loss.item() * current_batch_size)

                loss.backward()
                self.optimizer.step()

            # log.info(f"Epoch {str(epoch)} took {str(round(stop - start))} seconds \n")
            # Track epoch loss
            loss_per_epoch.append(current_epoch_loss / len(pos_triples))

        stop_training = timeit.default_timer()
        log.info(f"Training took {str(round(stop_training - start_training))} seconds \n")

        return loss_per_epoch


def slice_triples(triples: np.ndarray):
    """Get the heads, relations, and tails from a matrix of triples."""
    h = triples[:, 0:1]
    r = triples[:, 1:2]
    t = triples[:, 2:3]
    return h, r, t

def _split_list_in_batches(input_list: np.ndarray,
                           batch_size: int):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]
