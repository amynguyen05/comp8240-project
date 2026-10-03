import torch
import torch.nn.functional as F
from fastopic._fastopic import fastopic
from fastopic._model_utils import pairwise_euclidean_distance


class FastopicNoETP(fastopic):
    def _theta(self, doc_embeddings, topic_embeddings):
        dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
        return F.softmax(-dist / self.theta_temp, dim=1)  # Eq. (2), over topics

    def _beta(self):
        dist = pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        return F.softmax(-dist / self.theta_temp, dim=1)  # Eq. (2), over words

    def get_beta(self):
        with torch.no_grad():
            return self._beta()

    def get_theta(self, doc_embeddings, train_doc_embeddings):
        with torch.no_grad():
            topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
            return self._theta(doc_embeddings, topic_embeddings)

    def forward(self, train_bow, doc_embeddings):
        theta = self._theta(doc_embeddings, self.topic_embeddings)
        beta = self._beta()
        recon = torch.matmul(theta, beta)
        loss = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()
        return {"loss": loss}
