# SPDX-FileCopyrightText: 2021-2022 Simon Reiss, Karlsruher Institut für Technologie (KIT) and Carl Zeiss AG
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
import numpy as np


def con2r_loss(embeddings, predictions, sampling_number, receptive_volume_size, alpha):
    """
    This function processes volumes in a batch sequentially by first setting up a target-similarity graph,
    then computing the current similarity graph from voxel-embeddings and then aligns them to the target graph.

    Args:
        embeddings: Embedding tensor of shape batch_size x embedding_dim x depth x width x height
        predictions: Softmaxed prediction tensor of shape batch_size x num_classes x depth x width x height
        sampling_number: Number of voxels that get sampled in the query and neighbor set
        receptive_volume_size: Tuple containing the size of receptive volumes (r_depth, r_width, r_height)
        alpha: interpolation hyperparameter between positional smoothness and semantic coherence constraints

    Returns: Con2R loss as tensor for back-propagation

    """
    epsilon = 1e-7
    device = embeddings.device
    batch_size = embeddings.shape[0]

    loss_con2r = 0.0

    # Process volumes sequentially
    for b_idx in range(batch_size):
        # Set the current voxel-embeddings and voxel-predictions
        current_emb = embeddings[b_idx]
        current_preds = predictions[b_idx]
        shape = current_emb.shape

        # Sample position in the volume for the query embeddings
        query_d = torch.empty(sampling_number, dtype=torch.long).random_(shape[1]).to(device)
        query_w = torch.empty(sampling_number, dtype=torch.long).random_(shape[2]).to(device)
        query_h = torch.empty(sampling_number, dtype=torch.long).random_(shape[3]).to(device)

        # Sample position in the volume for the neighbor embeddings
        neighbor_d = torch.empty(sampling_number, dtype=torch.long).random_(shape[1]).to(device)
        neighbor_w = torch.empty(sampling_number, dtype=torch.long).random_(shape[2]).to(device)
        neighbor_h = torch.empty(sampling_number, dtype=torch.long).random_(shape[3]).to(device)

        # Select query and neighbor embeddings
        query_emb = current_emb[:, query_d, query_w, query_h].permute(1, 0)
        neighbor_emb = current_emb[:, neighbor_d, neighbor_w, neighbor_h].permute(1, 0)

        # Query and neighbor coordinates/positions
        query_coordinates = torch.stack([query_d, query_w, query_h]).unsqueeze(-1).to(device).float()
        neighbor_coordinates = torch.stack([neighbor_d, neighbor_w, neighbor_h]).unsqueeze(-1).to(device).float()

        # Compute and positional smoothness constraint based on receptive volume intersections
        depth_dist = torch.cdist(query_coordinates[0], neighbor_coordinates[0], p=1).unsqueeze(0)
        width_dist = torch.cdist(query_coordinates[1], neighbor_coordinates[1], p=1).unsqueeze(0)
        height_dist = torch.cdist(query_coordinates[2], neighbor_coordinates[2], p=1).unsqueeze(0)

        complete_dist = torch.cat([depth_dist, width_dist, height_dist], dim=0)

        complete_dist[0][complete_dist[0] > receptive_volume_size[0]] = receptive_volume_size[0]
        complete_dist[0] = receptive_volume_size[0] - complete_dist[0]

        complete_dist[1][complete_dist[1] > receptive_volume_size[1]] = receptive_volume_size[1]
        complete_dist[1] = receptive_volume_size[1] - complete_dist[1]

        complete_dist[2][complete_dist[2] > receptive_volume_size[2]] = receptive_volume_size[2]
        complete_dist[2] = receptive_volume_size[2] - complete_dist[2]

        positional_smoothness = complete_dist.prod(dim=0) / (np.prod(receptive_volume_size))

        positional_smoothness[positional_smoothness <= epsilon] = epsilon

        # Normalize row- and column-wise
        positional_smoothness_2 = positional_smoothness / positional_smoothness.sum(dim=0, keepdim=True)
        positional_smoothness = positional_smoothness / positional_smoothness.sum(dim=1, keepdim=True)

        # Compute semantic coherence constraint
        query_preds = current_preds[:, query_d, query_w, query_h].permute(1, 0)
        neighbor_preds = current_preds[:, neighbor_d, neighbor_w, neighbor_h].permute(1, 0)

        # Semantic coherence (symmetrized negative KL-Divergence), epsilon for numeric stability
        neg_sym_kl = neighbor_preds.unsqueeze(1) * (neighbor_preds.unsqueeze(1).div(
            query_preds.unsqueeze(0) + epsilon) + epsilon).log()
        neg_sym_kl_2 = query_preds.unsqueeze(1) * (query_preds.unsqueeze(1).div(
            neighbor_preds.unsqueeze(0) + epsilon) + epsilon).log()
        sym = torch.exp(-(0.5 * neg_sym_kl.sum(dim=2) + 0.5 * neg_sym_kl_2.sum(dim=2)))

        # Normalize row- and column-wise
        semantic_coherence = sym / (sym.sum(dim=1, keepdim=True) + epsilon)
        semantic_coherence_2 = sym / (sym.sum(dim=0, keepdim=True) + epsilon)

        # Setup target-similarity graph
        target_graph = alpha * positional_smoothness + (1 - alpha) * semantic_coherence
        target_graph_2 = alpha * positional_smoothness_2 + (1 - alpha) * semantic_coherence_2

        # Compute similarity graph from the voxel-embeddings, normalize and align to target graph
        embedding_graph = torch.exp(torch.mm(query_emb, neighbor_emb.detach().t()))
        embedding_graph = embedding_graph / embedding_graph.sum(dim=1, keepdim=True)
        loss_con2r_1 = - (torch.log(embedding_graph + epsilon) * target_graph).sum(1).mean()

        # Compute similarity graph from the voxel-embeddings, normalize and align to target graph
        # (query and neighbor set switched)
        embedding_graph_2 = torch.exp(torch.mm(query_emb.detach(), neighbor_emb.t()))
        embedding_graph_2 = embedding_graph_2 / embedding_graph_2.sum(dim=0, keepdim=True)
        loss_con2r_2 = - (torch.log(embedding_graph_2 + epsilon) * target_graph_2).sum(0).mean()

        loss_con2r += 0.5 * loss_con2r_1 + 0.5 * loss_con2r_2
    return loss_con2r / batch_size
