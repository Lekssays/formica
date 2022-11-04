import torch

import utils

class Aggregator:
    def __init__(self):
        config = utils.get_config()
        self.num_entities = config["metadata"]["num_entities"]
        self.model_name = config["model"]
        self.hidden_dim = config["hidden_dim"]
        self.device = torch.device(config["device"])
        self.aggregation_mode = config["aggregation_mode"]

    def fede_aggregate(self, state_dicts):
        ent_freq_list = []
        for state_dict in state_dicts:
            ent_freq_list.append(state_dict["entity_freq"])

        ent_freq_mat = torch.stack(ent_freq_list).to(self.device)
        agg_ent_mask = torch.stack(ent_freq_list).to(self.device)
        agg_ent_mask[ent_freq_mat != 0] = 1

        ent_w_sum = torch.sum(agg_ent_mask, dim=0) # freq of entities in all data
        ent_w = agg_ent_mask / ent_w_sum # shape=(n_clients,n_entities)
        ent_w[torch.isnan(ent_w)] = 0

        if self.model_name in ['RotatE', 'ComplEx']:
            update_ent_embed = torch.zeros(self.num_entities, self.hidden_dim * 2).to(self.device)
        else:
            update_ent_embed = torch.zeros(self.num_entities, self.hidden_dim).to(self.device)

        for i, state_dict in enumerate(state_dicts):
            ent_embed = state_dict["entity_embedding"]
            update_ent_embed += ent_embed * ent_w[i].reshape(-1, 1)

        # new_entity_embedding = update_ent_embed.requires_grad_()

        aggregated_state_dict = {
            "entity_embedding": update_ent_embed,
            "relation_embedding": state_dicts[-1]["relation_embedding"],
            "entity_freq": state_dicts[-1]["relation_embedding"],
        }

        return aggregated_state_dict

    def isolation_aggregate(self, state_dicts):
        return state_dicts[-1]

    def __call__(self, state_dicts):
        print("Do ", self.aggregation_mode)
        if self.aggregation_mode == "FedE":
            return self.fede_aggregate(state_dicts)
        elif self.aggregation_mode == "Isolation":
            return self.isolation_aggregate(state_dicts)
        else:
            raise Exception("Aggregation mode: {} is unsupported".format(self.aggregation_mode))
