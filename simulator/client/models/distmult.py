import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_name = "DistMult"

    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        if not neg:
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        else:
            head_part, tail_part = sample
            batch_size = head_part.shape[0]

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part == None:
                tail = entity_embedding.unsqueeze(0)
            else:
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
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

    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score