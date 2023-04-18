# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from dataclasses import dataclass

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer

from comet import download_model, load_from_checkpoint


@dataclass
class COMETScorerConfig(FairseqDataclass):
    pass


@register_scorer("comet", dataclass=COMETScorerConfig)
class COMETScorer(BaseScorer):
    def __init__(self, args):
        super(COMETScorer, self).__init__(args)
        
        self.model_path = download_model("Unbabel/wmt22-comet-da")
        self.model = load_from_checkpoint(self.model_path)

    def score(self, src, mt, ref):
        """
        Method to score an NMT using the COMET model architecture
        
        Args:
        - src [string]: The source sentence to be translated
        - mt [string]: The machine translated sentence
        - ref [string]: The target "true" translation
        """
        self.translation_score = self.model.predict({"src": src, "mt": mt, "ref": ref})
        return self.translation_score

    def result_string(self, order=4):
        return f"COMET: {self.score():.4f}"
