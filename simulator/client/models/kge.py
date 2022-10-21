import torch
import torch.nn as nn
import os

class KGEModel(nn.Module):
    def __init__(self, model_id, model_name, args):
        super(KGEModel, self).__init__()
        self.model_id = model_id
        self.model_name = model_name
        self.num_entities = args["num_entities"]
        self.num_relations = args["num_relations"]
        self.device = torch.device(args["device"])

        self.gamma = nn.Parameter(
            torch.Tensor([args["gamma"]]),
            requires_grad=False
        )

        self.embedding_range = torch.Tensor([(args["gamma"] + args["epsilon"]) / args["hidden_dim"]])

        if model_name in ['RotatE', 'ComplEx']:
            self.entity_embedding = torch.zeros(self.num_entities, args["hidden_dim"]*2).to(self.device).requires_grad_()
        else:
            self.entity_embedding = torch.zeros(self.num_entities, args["hidden_dim"]).to(self.device).requires_grad_()

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name in ['ComplEx']:
            self.relation_embedding = torch.zeros(self.num_relations, args["hidden_dim"] * 2).to(self.device).requires_grad_()
        else:
            self.relation_embedding = torch.zeros(self.num_relations, args["hidden_dim"]).to(self.device).requires_grad_()

        nn.init.uniform_(
            tensor=self.self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def get_state_dict(self):
        state = {
            'entity_embedding': self.entity_embedding,
            'relation_embedding': self.relation_embedding
        }

        return state

    def set_state_dict(self, state_dict):
        self.entity_embedding = state_dict["entity_embedding"]
        self.relation_embedding = state_dict["relation_embedding"]

    def save_model(self, state_dir):
        state_dict = self.get_state_dict()

        state_path = os.path.join(state_dir, "{}.pt".format(self.model_id))

        torch.save(state_dict, state_path)

    def load_model(self, state_dir):
        state_path = os.path.join(state_dir, "{}.pt".format(self.model_id))
        state_dict = torch.load(state_path, map_location=self.device)
        self.set_state_dict(state_dict)

    def forward(self, sample, neg=True):
        if not neg:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        else:
            head_part, tail_part = sample
            batch_size = head_part.shape[0]

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part == None:
                tail = self.entity_embedding.unsqueeze(0)
            else:
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }

        score = model_func[self.model_name](head, relation, tail)

        return score

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
