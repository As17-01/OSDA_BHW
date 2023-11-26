from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from sklearn.metrics import roc_auc_score

import neural_fca_example.neural_lib as nl


class FormalConcept:
    def __init__(self, thr_dict: Dict[str, Any]):
        self.thr_dict = thr_dict
        self.cn = None

    def fit(self, data: pd.DataFrame, target: pd.Series):
        k_train = FormalContext.from_pandas(data)

        lattice = ConceptLattice.from_context(k_train, is_monotone=True)

        for c in lattice:
            y_preds = np.zeros(k_train.n_objects)
            y_preds[list(c.extent_i)] = 1
            c.measures = c.measures.set("f1_score", roc_auc_score(target, y_preds))

        best_concepts = list(lattice.measures["f1_score"].argsort()[::-1][:7])

        self.cn = nl.ConceptNetwork.from_lattice(lattice, best_concepts, sorted(set(target)))
        self.cn.fit(data, target)

    def predict_proba(self, data: pd.DataFrame):
        predictions = self.cn.predict_proba(data).detach().numpy()
        return predictions
