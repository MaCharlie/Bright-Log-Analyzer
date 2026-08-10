from sentence_transformers import CrossEncoder

import config


class Reranker:
    def __init__(self):
        self.reranker_model = CrossEncoder(config.reranker_model)

    def rerank_results(self, query, candidates):
        scores = self.reranker_model.predict([(query, candidate) for candidate in candidates])
        return sorted(zip(candidates, scores), key=lambda c: c[1], reverse=True)