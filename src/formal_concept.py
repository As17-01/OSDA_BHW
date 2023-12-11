from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
import torch
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from sklearn.metrics import accuracy_score

import neural_fca_example.neural_lib as nl


class FormalConcept:
    def __init__(
        self,
        thr_dict: Dict[str, Any],
        n_concepts: int,
        loss_fn=torch.nn.CrossEntropyLoss(),
        nonlinearity=torch.nn.ReLU,
        n_epochs=10,
    ):
        self.thr_dict = thr_dict
        self.n_concepts = n_concepts
        self.cn = None
        self.loss_fn = (loss_fn,)
        self.nonlinearity = nonlinearity
        self.n_epochs = n_epochs

    def fit(self, data: pd.DataFrame, target: pd.Series):
        k_train = FormalContext.from_pandas(data)

        lattice = ConceptLattice.from_context(k_train, is_monotone=True)

        for c in lattice:
            y_preds = np.zeros(k_train.n_objects)
            y_preds[list(c.extent_i)] = 1
            c.measures = c.measures.set("accuracy", accuracy_score(target, y_preds))

        best_concepts = list(lattice.measures["accuracy"].argsort()[::-1][: self.n_concepts])

        self.cn = nl.ConceptNetwork.from_lattice(lattice, best_concepts, sorted(set(target)))
        self.cn.fit(
            data, pd.Series(target), loss_fn=self.loss_fn, nonlinearity=self.nonlinearity, n_epochs=self.n_epochs
        )

    def predict_proba(self, data: pd.DataFrame):
        predictions = self.cn.predict_proba(data).detach().numpy()
        return predictions
