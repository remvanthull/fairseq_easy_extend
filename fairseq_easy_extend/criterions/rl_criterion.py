import math
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor

from dataclasses import dataclass, field
import sacrebleu
from bert_score import score as bert_score

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tgt_dict = task.target_dictionary
        self.tgt_lang = "en"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _compute_loss(self, outputs, targets, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """

        # input to device
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)
        
        # B x T x V
        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)

        # sample predictions and convert predictions and targets to strings
        with torch.no_grad():
            probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
            predicted  = torch.multinomial(probs, 1,replacement=True).view(bsz, seq_len)
            predicted_str = [self.tgt_dict.string(pred) for pred in predicted]
            target_str = [self.tgt_dict.string(target) for target in targets]
        
        # calculate metric score
        with torch.no_grad():
            if self.metric == "bleu":
                score = torch.tensor([[sacrebleu.sentence_bleu(pred, [targ]).score] * seq_len for pred, targ in zip(predicted_str, target_str)])
                # score = sacrebleu.sentence_bleu(predicted_str, [target_str]).score
            elif self.metric == "chrf":
                score = torch.tensor([[sacrebleu.sentence_chrf(pred, [targ]).score] * seq_len for pred, targ in zip(predicted_str, target_str)])
            elif self.metric == "bert":
                _, _, score = bert_score([predicted_str], [target_str], lang=self.tgt_lang)
                score = torch.tensor([[bert_score([pred], [targ], lang=self.tgt_lang).score[2].mean()] * seq_len for pred, targ in zip(predicted_str, target_str)])
        
        # take masks
        if masks is not None:
            masks = masks.to(self.device)
            outputs, targets = outputs[masks], targets[masks]
            score, predicted = score[masks], predicted[masks]

        # get the log probs of the samples
        log_probs = F.log_softmax(outputs, dim=-1)
        score = score.to(self.device)
        sample_log_probs = torch.gather(log_probs, 1, predicted.unsqueeze(1)).squeeze()
        
        # calculate loss of all samples and average for batch loss
        loss = -sample_log_probs * score
        loss = loss.mean()
        print(loss)
        return loss
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        #get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss = self._compute_loss(outs, tgt_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True