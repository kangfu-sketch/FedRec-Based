import torch
from engine import Engine


class MLP(torch.nn.Module):
    def __init__(self, config, pre_trained_embeddings=None):
        super(MLP, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if config['pre_embedding'] and pre_trained_embeddings != None:
            self.embedding_item.weight.data = pre_trained_embeddings
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.init_weight()

    def forward(self, item_indices):
        item_embedding = self.embedding_item(item_indices)
        logits = self.affine_output(item_embedding)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        """实际的权重初始化"""
        if self.embedding_item is not None:
            torch.nn.init.xavier_uniform_(self.embedding_item.weight)


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config, pre_trained_embeddings=None):
        self.model = MLP(config, pre_trained_embeddings)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
