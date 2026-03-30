import pytest
import pandas as pd
import numpy as np

# TODO: Import your sampler functions once implemented
# from master_thesis.episode_sampler import DepartmentEpisodeSampler

def test_sampler_no_data_leakage():
    """
    Guarantee that the support and query sets have no overlapping samples,
    preventing data leakage during meta-training.
    """
    # sampler = DepartmentEpisodeSampler(...)
    # support_set, query_set = sampler.sample_episode(department="Logistics")
    
    # support_ids = set(support_set['contract_id'])
    # query_ids = set(query_set['contract_id'])
    
    # intersection = support_ids.intersection(query_ids)
    # assert len(intersection) == 0, f"Found data leakage! Overlapping IDs: {intersection}"
    pass

def test_n_way_k_shot_stratification():
    """
    Ensure the sampled support set has exactly K positive and K negative examples
    (for a 2-way K-shot binary classification setup).
    """
    # k_shots = 5
    # sampler = DepartmentEpisodeSampler(k_shots=k_shots)
    # support_set, _ = sampler.sample_episode(department="Logistics")
    
    # pos_count = (support_set['label'] == 1).sum()
    # neg_count = (support_set['label'] == 0).sum()
    
    # assert pos_count == k_shots, f"Expected {k_shots} positive samples, got {pos_count}"
    # assert neg_count == k_shots, f"Expected {k_shots} negative samples, got {neg_count}"
    pass
