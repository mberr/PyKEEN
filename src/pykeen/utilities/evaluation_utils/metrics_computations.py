# -*- coding: utf-8 -*-

"""Script to compute mean rank and hits@k."""

import logging
import timeit
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ...constants import EMOJI

log = logging.getLogger(__name__)

DEFAULT_HITS_AT_K = [1, 3, 5, 10]


def _hash_triples(triples: Iterable[Hashable]) -> int:
    """Hash a list of triples."""
    return hash(tuple(triples))


def update_hits_at_k(
        hits_at_k_values: Dict[int, List[float]],
        rank_of_positive_subject_based: int,
        rank_of_positive_object_based: int
) -> None:
    """Update the Hits@K dictionary for two values."""
    for k, values in hits_at_k_values.items():
        if rank_of_positive_subject_based <= k:
            values.append(1.0)
        else:
            values.append(0.0)

        if rank_of_positive_object_based <= k:
            values.append(1.0)
        else:
            values.append(0.0)

def update_multiple_hits_at_k(
        hits_at_k_values: Dict[int, List[float]],
        ranks_of_positive_subjects_based: List[int],
        ranks_of_positive_objects_based: List[int]
) -> None:
    """Update the Hits@K dictionary for two values."""
    for k, values in hits_at_k_values.items():
        for rank_of_positive_subject_based in ranks_of_positive_subjects_based:
            if rank_of_positive_subject_based <= k:
                values.append(1.0)
            else:
                values.append(0.0)

        for rank_of_positive_object_based in ranks_of_positive_objects_based:
            if rank_of_positive_object_based <= k:
                values.append(1.0)
            else:
                values.append(0.0)


def _filter_corrupted_triples(
        pos_triple,
        subject_batch,
        object_batch,
        device,
        all_pos_triples,
):
    subject = pos_triple[0:1]
    relation = pos_triple[1:2]
    object = pos_triple[2:3]

    subject_filter = all_pos_triples[:,0:1] == subject
    relation_filter = all_pos_triples[:, 1:2] == relation
    object_filter = all_pos_triples[:, 2:3] == object

    # Short objects batch list
    filter = (subject_filter & relation_filter)
    objects_in_triples = all_pos_triples[:, 2:3][filter]
    object_batch[objects_in_triples] = False

    # Short subjects batch list
    filter = (object_filter & relation_filter)
    subjects_in_triples = all_pos_triples[:, 0:1][filter]
    subject_batch[subjects_in_triples] = False

    # TODO: Create warning when all triples will be filtered
    # if mask.size == 0:
    #     raise Exception("User selected filtered metric computation, but all corrupted triples exists"
    #                     "also as positive triples.")

    return subject_batch, object_batch


def _compute_filtered_rank(
        kg_embedding_model,
        pos_triple,
        subject_batch,
        object_batch,
        device,
        all_pos_triples,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed:
    """
    subject_batch, object_batch = _filter_corrupted_triples(
        pos_triple=pos_triple,
        subject_batch = subject_batch,
        object_batch = object_batch,
        device=device,
        all_pos_triples=all_pos_triples,
    )

    return _compute_rank(
        kg_embedding_model=kg_embedding_model,
        pos_triple=pos_triple,
        subject_batch=subject_batch,
        object_batch=object_batch,
        device=device,
        all_pos_triples=all_pos_triples,
    )


def _compute_rank(
        kg_embedding_model,
        pos_triple,
        subject_batch,
        object_batch,
        device,
        all_pos_triples,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param pos_triple:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed: This parameter isn't used but is necessary for compatibility
    """
    subject = pos_triple[0:1]
    relation = pos_triple[1:2]
    object = pos_triple[2:3]

    inverse_triple = torch.cat((object,
                                relation + kg_embedding_model.inverse_model * kg_embedding_model.num_relations //2,
                                subject))

    scores_of_corrupted_subjects = kg_embedding_model.predict_for_ranking(object,
                                                                          relation + kg_embedding_model.inverse_model *
                                                                          kg_embedding_model.num_relations // 2)

    score_of_positive_subject = scores_of_corrupted_subjects[subject]
    scores_of_corrupted_subjects = scores_of_corrupted_subjects[subject_batch]

    scores_of_corrupted_objects = kg_embedding_model.predict_for_ranking(subject, relation)
    score_of_positive_object = scores_of_corrupted_objects[object]
    scores_of_corrupted_objects = scores_of_corrupted_objects[object_batch]

    rank_of_positive_subject_based = (scores_of_corrupted_subjects > score_of_positive_subject).sum() + 1

    rank_of_positive_object_based = (scores_of_corrupted_objects > score_of_positive_object).sum() + 1

    return (
        rank_of_positive_subject_based.detach().cpu().numpy(),
        rank_of_positive_object_based.detach().cpu().numpy(),
    )


@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    mean_reciprocal_rank: float
    hits_at_k: Dict[int, float]


def compute_metric_results(
        kg_embedding_model,
        mapped_train_triples,
        mapped_test_triples,
        device,
        filter_neg_triples=False,
        ks: Optional[List[int]] = None,
        *,
        use_tqdm: bool = True,
) -> MetricResults:
    """Compute the metric results.

    :param all_entities:
    :param kg_embedding_model:
    :param mapped_train_triples:
    :param mapped_test_triples:
    :param device:
    :param filter_neg_triples:
    :param ks:
    :param use_tqdm: Should a progress bar be shown?
    :return:
    """
    start = timeit.default_timer()

    ranks: List[int] = []
    ranks_of_positive_subjects_based = []
    ranks_of_positive_objects_based = []
    hits_at_k_values = {
        k: []
        for k in (ks or DEFAULT_HITS_AT_K)
    }
    kg_embedding_model = kg_embedding_model.eval()
    kg_embedding_model = kg_embedding_model.to(device)

    all_pos_triples = np.concatenate([mapped_train_triples, mapped_test_triples], axis=0)
    all_pos_triples = torch.tensor(all_pos_triples, device=device)
    all_entities = torch.arange(kg_embedding_model.num_entities, device=device)

    compute_rank_fct: Callable[..., Tuple[int, int]] = (
        _compute_filtered_rank
        if filter_neg_triples else
        _compute_rank
    )

    mapped_test_triples = torch.tensor(mapped_test_triples, dtype=torch.long, device=device)

    mapped_test_triples = mapped_test_triples[(mapped_test_triples[:,1:2].flatten()).argsort()]

    weird_triples = []

    if use_tqdm:
        mapped_test_triples = tqdm(mapped_test_triples, desc=f'{EMOJI} corrupting triples')

    for pos_triple in mapped_test_triples:
        subject = pos_triple[0:1]
        object = pos_triple[2:3]

        subject_batch = all_entities != subject
        object_batch = all_entities != object

        rank_of_positive_subject_based, rank_of_positive_object_based = compute_rank_fct(
            kg_embedding_model=kg_embedding_model,
            pos_triple=pos_triple,
            subject_batch=subject_batch,
            object_batch=object_batch,
            device=device,
            all_pos_triples=all_pos_triples,
        )
        # log.info(f"Calculating scores took {timeit.default_timer() - start:.3f} seconds")

        ranks.append(rank_of_positive_subject_based)
        ranks.append(rank_of_positive_object_based)
        ranks_of_positive_objects_based.append(rank_of_positive_object_based)
        ranks_of_positive_subjects_based.append(rank_of_positive_subject_based)


    # Compute hits@k for k in {1,3,5,10}
    update_multiple_hits_at_k(
        hits_at_k_values,
        ranks_of_positive_subjects_based=ranks_of_positive_subjects_based,
        ranks_of_positive_objects_based=ranks_of_positive_objects_based,
    )

    mean_rank = float(np.mean(ranks))
    mean_reciprocal_rank = float((1/np.vstack(ranks)).mean())
    hits_at_k: Dict[int, float] = {
        k: np.mean(values)
        for k, values in hits_at_k_values.items()
    }

    stop = timeit.default_timer()
    log.info(f"Evaluation took {stop-start:.2f} seconds")

    return MetricResults(
        mean_rank=mean_rank,
        mean_reciprocal_rank=mean_reciprocal_rank,
        hits_at_k=hits_at_k,
    )
