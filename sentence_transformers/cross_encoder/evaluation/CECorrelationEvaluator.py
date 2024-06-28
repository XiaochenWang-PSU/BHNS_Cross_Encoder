import logging
from scipy.stats import pearsonr, spearmanr
from typing import List
import os
import csv
from ... import InputExample
from sklearn.metrics import ndcg_score,roc_curve, auc
import numpy as np

logger = logging.getLogger(__name__)

class CECorrelationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and continuous scores,
    it compute the pearson & spearman correlation between the predicted score for the sentence pair
    and the gold score.
    """
    def __init__(self, sentence_pairs: List[List[str]], scores: List[float], name: str='', write_csv: bool = True):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.name = name

        self.csv_file = "CECorrelationEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Pearson_Correlation", "Spearman_Correlation"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples, **kwargs):
        sentence_pairs = []
        scores = []

        for example in examples:
            sentence_pairs.append(example.texts)
            scores.append(example.label)
        return cls(sentence_pairs, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CECorrelationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

        
        print(len(self.scores), len(pred_scores))
        eval_pearson, _ = pearsonr(self.scores, pred_scores)
        eval_spearman, _ = spearmanr(self.scores, pred_scores)
        fpr, tpr, thresholds = roc_curve([1 if i>=0.5 else 0 for i in self.scores], pred_scores)
        auc_score = auc(fpr, tpr)
        y_scores = np.asarray([self.scores])
        p_scores = np.asarray([pred_scores])
        


        logger.info("Correlation:\tPearson: {:.4f}\tSpearman: {:.4f}\tAUC: {:.4f}\t".format(eval_pearson, eval_spearman, auc_score))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson, eval_spearman])

        return eval_spearman
