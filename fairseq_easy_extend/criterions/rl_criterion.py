
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

import torch


from dataclasses import dataclass, field

import sacrebleu
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import CHRF

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})




@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        #padding mask, do not remove
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
        

        #we take a softmax over outputs
        #argmax over the softmax \ sampling (e.g. multinomial)
        #sampled_sentence = [4, 17, 18, 19, 20]
        #sampled_sentence_string = tgt_dict.string([4, 17, 18, 19, 20])
        #see dictionary class of fairseq
        #target_sentence = "I am a sentence"
        #with torch.no_grad()
            #R(*) = eval_metric(sampled_sentence_string, target_sentence)
            #R(*) is a number, BLEU, сhrf, etc.

        #loss = -log_prob(outputs)*R()
        #loss = loss.mean()

        with torch.no_grad():
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            # Sample from the multinomial distribution
            dists = torch.distributions.Categorical(logits=log_probs)
            predicted = dists.sample()
            predicted_str = self.target_dictionary.string(predicted)
            if self.metric == "bleu":
                score = sacrebleu.corpus_bleu(predicted_str, targets).score
            elif self.metric == "chrf":
                score = sacrebleu.sentence_chrf(predicted_str, targets).score
        
        loss = -log_probs * score
        loss = loss.mean() * factor

        return loss