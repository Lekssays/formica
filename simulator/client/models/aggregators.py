import torch

import utils

class FedEAggregator:
    def __init__(self):
        config = utils.get_config()
        self.num_entities = config["num_entities"]
        self.model_name = config["model"]
        self.hidden_dim = config["hidden_dim"]

    def __call__(self, state_dicts, ent_freq_mat):
        agg_ent_mask = ent_freq_mat
        agg_ent_mask[ent_freq_mat != 0] = 1

        ent_w_sum = torch.sum(agg_ent_mask, dim=0) # freq of entities in all data
        ent_w = agg_ent_mask / ent_w_sum # shape=(n_clients,n_entities)
        ent_w[torch.isnan(ent_w)] = 0

        if self.model_name in ['RotatE', 'ComplEx']:
            update_ent_embed = torch.zeros(self.num_entities, self.hidden_dim * 2)
        else:
            update_ent_embed = torch.zeros(self.num_entities, self.hidden_dim)

        for i, state_dict in enumerate(state_dicts):
            ent_embed = state_dict["entity_embedding"]
            update_ent_embed += ent_embed * ent_w[i].reshape(-1, 1)

        new_entity_embedding = update_ent_embed.requires_grad_()

        return new_entity_embedding