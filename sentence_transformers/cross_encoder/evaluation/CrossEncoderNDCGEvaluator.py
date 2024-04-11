import logging
import os
import csv 
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from typing import List
#from evaluation import NDCGEvaluator
#import NDCGEvaluator

logger = logging.getLogger(__name__)


from sentence_transformers.evaluation import SentenceEvaluator
import logging
import os
import csv 
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from typing import List


logger = logging.getLogger(__name__)

class NDCGEvaluator(SentenceEvaluator):
    """ 
    Evaluate a model based on the similarity of the embeddings by calculating NDCG
    in comparison to the gold standard labels.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float],
                 df: pd.DataFrame,
                 offline_scores: List[float] = [],
                 batch_size: int = 16, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """ 
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1: List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: List of similarity scores between sentences1 and sentences2.
                       This can be either binary or scaled scores.
                       For binary version, possible scores are in [0, 1]
                       For scaled version, possible scores are in [0, 1, 2, 3]
                       where 3 for 'strongly relevant', 2 for 'relevant' 1 for 'somewhat relevant'
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = np.array(scores)
        # We treat all the scores > 0 to be positive labels for AUC/accuracy/log loss calculation
        self.labels = np.array(self.scores > 0, dtype=int)
        self.write_csv = write_csv
        self.df = df
        self.num_search_ids = len(df['search_id'].unique())
        self.offline_scores = offline_scores

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20", "auc", "accuracy", "log_loss"]

        logger.info("{}: Num distinct search_ids for {} dataset: {}".format(self.__class__.__name__, self.name, str(self.num_search_ids)))


    def from_input_examples(cls, examples: List[object], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        search_ids = {}
        distinct_ids = 0
        data = []
        offline_scores = []

        for example in examples:
            term = example.texts[0]
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)

            if term not in search_ids:
                search_ids[term] = distinct_ids
                distinct_ids += 1
            search_id = search_ids[term]

            data.append([search_id, term, example.texts[1], example.label])

            # add offline_score if exists
            if hasattr(example, 'offline_score') and example.offline_score:
                offline_scores.append(example.offline_score)

        df = pd.DataFrame(data, columns = ['search_id', 'term', 'product', 'converted'])

        return cls(sentences1, sentences2, scores, df, offline_scores, **kwargs)

    def sigmoid(self, x):
        """
        Get sigmoid.
        """
        return 1 / (1 + np.exp(-x))

    def dcg_at_k(self, r, k, method=0):
        """Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Example from
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
        :param r: list of relevance scores sorted by the model prediction scores
        :param k: the k in top-k
        :param method: 0 means DCG = r[0] + r[1]/log2(2) + r[2]/log2(3) + ... r[n]/log2(n+1)
                       1 means DCG = r[0]/log2(2) + r[1]/log2(3) + ... r[n]/log2(n+2)
                       method 1 emphasizes on retrieving highly relevant documents
                       since method 1 differentiates gains for r[0] and r[1] while
                       method 0 doesn't differentiate gains for r[0] and r[1]
                       (r[1] = r[1]/log2(2))
        Returns:
            Discounted cumulative gain
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.
    
    def ndcg_at_k(self, r, k, method=0):
        """Score is normalized discounted cumulative gain (ndcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Example from                                                                                            
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
        Returns:
            Normalized discounted cumulative gain
        """
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def ndcg_tot(self, y_pred=None, k=1, show_individual_ndcg=False):
        prediction = pd.DataFrame()
        prediction = self.df.loc[:, ['search_id', 'term', 'converted']]
        if y_pred is not None:
            prediction.loc[:, 'pred'] = y_pred
        else:
            prediction['pred'] = self.df['converted']
        prediction = prediction.sort_values(['search_id', 'pred'], ascending=False)
        groups = prediction.groupby(['search_id'])
        i = 0
        dcg_sum = 0
        ndcg_sum = 0
        rank_sum = 0
        ndcgs = []
        for index, group in groups:
            ndcg_tmp = self.ndcg_at_k(list(group.loc[:, 'converted']), k)
            ndcg_sum += ndcg_tmp
            term = group.iloc[0]['term']
            ndcgs.append((term, ndcg_tmp))
            i = i + 1
    
        if show_individual_ndcg:
            ndcgs.sort(key=lambda x: x[1])
            logger.info("Top 100 worst performing queries")
            for term, ndcg_tmp in ndcgs[0:100]:
                logger.info("{} : {:.4f}".format(term, ndcg_tmp))

        ndcg = ndcg_sum / i
        return (ndcg)

    def recall_k(self, y_pred=None, k=1):
        prediction = pd.DataFrame()
        prediction = self.df.loc[:, ['search_id', 'term', 'converted']]
        if y_pred is not None:
            prediction.loc[:, 'pred'] = y_pred
        else:
            prediction['pred'] = self.df['converted']
        prediction = prediction.sort_values(['search_id', 'pred'], ascending=False)
        groups = prediction.groupby(['search_id'])
        i = 0
        recall_sum = 0

        for index, group in groups:
            # we use x > 0 since group['converted'] can either be binary or scaled values.
            scores = [float(x > 0) for x in group["converted"].tolist()]
            max_pos = int(sum(scores))
            if max_pos == 0:
                continue
            scores = scores[:min(k, max_pos)]
            l = len(scores)
            recall_tmp = sum(scores)/float(l)
            recall_sum += recall_tmp
            term = group.iloc[0]['term']
            i = i + 1

        recall = recall_sum / i
        return recall

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1,
            show_individual_ndcg: bool=False, normalize_embeddings: bool = False) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("{}: Evaluating the model on {} dataset{}".format(self.__class__.__name__, self.name, out_txt))

        if model is None:
            assert len(self.offline_scores) > 0
            score_proba = self.sigmoid(np.array(self.offline_scores))
            eval_mlc_loss = 0
        else:
            embeddings1 = model.encode(self.sentences1,
                    batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True,
                    normalize_embeddings=normalize_embeddings)
            embeddings2 = model.encode(self.sentences2,
                    batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True,
                    normalize_embeddings=normalize_embeddings)
            score_proba = self.sigmoid((embeddings1 * embeddings2).sum(-1))

        eval_ndcg5  = self.ndcg_tot(score_proba, k=5,  show_individual_ndcg=show_individual_ndcg)
        eval_ndcg10 = self.ndcg_tot(score_proba, k=10, show_individual_ndcg=show_individual_ndcg)
        eval_ndcg20 = self.ndcg_tot(score_proba, k=20, show_individual_ndcg=show_individual_ndcg)

        eval_recall5  = self.recall_k(score_proba, k=5)
        eval_recall10 = self.recall_k(score_proba, k=10)
        eval_recall20 = self.recall_k(score_proba, k=20)

        eval_auc = roc_auc_score(self.labels, score_proba)
        eval_accuracy = accuracy_score(self.labels, (score_proba > 0.5).astype(int))
        eval_log_loss = log_loss(self.labels, score_proba)

        logger.info(
            "NDCG@5: {:.4f}\tNDCG@10: {:.4f}\tNDCG@20: {:.4f}\tRECALL@5: {:.4f}\tRECALL@10: {:.4f}\tRECALL@20: {:.4f}\tAUC: {:.4f}\tAcc: {:.4f}\tLog Loss: {:.4f}".format(
            eval_ndcg5, eval_ndcg10, eval_ndcg20,
            eval_recall5, eval_recall10, eval_recall20,
            eval_auc, eval_accuracy, eval_log_loss))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_ndcg5, eval_ndcg10, eval_ndcg20,
                    eval_recall5, eval_recall10, eval_recall20,
                    eval_auc, eval_accuracy, eval_log_loss])

        return eval_ndcg20


class CrossEncoderNDCGEvaluator(SentenceEvaluator):
    """ 
    Evaluate a model based on the similarity of the embeddings by calculating NDCG
    in comparison to the gold standard labels.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float],
                 df: pd.DataFrame,
                 offline_scores: List[float] = [],
                 batch_size: int = 16, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """ 
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1: List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: List of similarity scores between sentences1 and sentences2.
                       This can be either binary or scaled scores.
                       For binary version, possible scores are in [0, 1]
                       For scaled version, possible scores are in [0, 1, 2, 3]
                       where 3 for 'strongly relevant', 2 for 'relevant' 1 for 'somewhat relevant'
        :param write_csv: Write results to a CSV file
        """
        super(CrossEncoderNDCGEvaluator, self).__init__(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=scores,
                df=df,
                offline_scores=offline_scores,
                batch_size=batch_size,
                name=name,
                show_progress_bar=show_progress_bar,
                write_csv=write_csv)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = np.array(scores)
        # We treat all the scores > 0 to be positive labels for AUC/accuracy/log loss calculation
        self.labels = np.array(self.scores > 0, dtype=int)
        self.write_csv = write_csv
        self.df = df
        self.num_search_ids = len(df['search_id'].unique())
        self.offline_scores = offline_scores

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20", "auc", "accuracy", "log_loss"]

        logger.info("{}: Num distinct search_ids for {} dataset: {}".format(self.__class__.__name__, self.name, str(self.num_search_ids)))

    def from_input_examples(cls, examples: List[object], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        search_ids = {}
        distinct_ids = 0
        data = []
        offline_scores = []

        for example in examples:
            term = example.texts[0]
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)

            if term not in search_ids:
                search_ids[term] = distinct_ids
                distinct_ids += 1
            search_id = search_ids[term]

            data.append([search_id, term, example.texts[1], example.label])

            # add offline_score if exists
            if hasattr(example, 'offline_score') and example.offline_score:
                offline_scores.append(example.offline_score)

        df = pd.DataFrame(data, columns = ['search_id', 'term', 'product', 'converted'])

        return cls(sentences1, sentences2, scores, df, offline_scores, **kwargs)

    def sigmoid(self, x):
        """
        Get sigmoid.
        """
        return 1 / (1 + np.exp(-x))

    def dcg_at_k(self, r, k, method=0):
        """Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Example from
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
        :param r: list of relevance scores sorted by the model prediction scores
        :param k: the k in top-k
        :param method: 0 means DCG = r[0] + r[1]/log2(2) + r[2]/log2(3) + ... r[n]/log2(n+1)
                       1 means DCG = r[0]/log2(2) + r[1]/log2(3) + ... r[n]/log2(n+2)
                       method 1 emphasizes on retrieving highly relevant documents
                       since method 1 differentiates gains for r[0] and r[1] while
                       method 0 doesn't differentiate gains for r[0] and r[1]
                       (r[1] = r[1]/log2(2))
        Returns:
            Discounted cumulative gain
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.
    
    def ndcg_at_k(self, r, k, method=0):
        """Score is normalized discounted cumulative gain (ndcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Example from                                                                                            
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
        Returns:
            Normalized discounted cumulative gain
        """
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def ndcg_tot(self, y_pred=None, k=1, show_individual_ndcg=False):
        prediction = pd.DataFrame()
        prediction = self.df.loc[:, ['search_id', 'term', 'converted']]
        if y_pred is not None:
            prediction.loc[:, 'pred'] = y_pred
        else:
            prediction['pred'] = self.df['converted']
        prediction = prediction.sort_values(['search_id', 'pred'], ascending=False)
        groups = prediction.groupby(['search_id'])
        i = 0
        dcg_sum = 0
        ndcg_sum = 0
        rank_sum = 0
        ndcgs = []
        for index, group in groups:
            ndcg_tmp = self.ndcg_at_k(list(group.loc[:, 'converted']), k)
            ndcg_sum += ndcg_tmp
            term = group.iloc[0]['term']
            ndcgs.append((term, ndcg_tmp))
            i = i + 1
    
        if show_individual_ndcg:
            ndcgs.sort(key=lambda x: x[1])
            logger.info("Top 100 worst performing queries")
            for term, ndcg_tmp in ndcgs[0:100]:
                logger.info("{} : {:.4f}".format(term, ndcg_tmp))

        ndcg = ndcg_sum / i
        return (ndcg)

    def recall_k(self, y_pred=None, k=1):
        prediction = pd.DataFrame()
        prediction = self.df.loc[:, ['search_id', 'term', 'converted']]
        if y_pred is not None:
            prediction.loc[:, 'pred'] = y_pred
        else:
            prediction['pred'] = self.df['converted']
        prediction = prediction.sort_values(['search_id', 'pred'], ascending=False)
        groups = prediction.groupby(['search_id'])
        i = 0
        recall_sum = 0

        for index, group in groups:
            # we use x > 0 since group['converted'] can either be binary or scaled values.
            scores = [float(x > 0) for x in group["converted"].tolist()]
            max_pos = int(sum(scores))
            if max_pos == 0:
                continue
            scores = scores[:min(k, max_pos)]
            l = len(scores)
            recall_tmp = sum(scores)/float(l)
            recall_sum += recall_tmp
            term = group.iloc[0]['term']
            i = i + 1

        recall = recall_sum / i
        return recall


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, show_individual_ndcg=False) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("{}: Evaluating the model on {} dataset{}".format(self.__class__.__name__, self.name, out_txt))

        model_input = [[query, doc] for query, doc in zip(self.sentences1, self.sentences2)]
        score_proba = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)

        eval_ndcg5  = self.ndcg_tot(score_proba, k=5,  show_individual_ndcg=show_individual_ndcg)
        eval_ndcg10 = self.ndcg_tot(score_proba, k=10, show_individual_ndcg=show_individual_ndcg)
        eval_ndcg20 = self.ndcg_tot(score_proba, k=20, show_individual_ndcg=show_individual_ndcg)

        eval_recall5  = self.recall_k(score_proba, k=5)
        eval_recall10 = self.recall_k(score_proba, k=10)
        eval_recall20 = self.recall_k(score_proba, k=20)

        eval_auc = roc_auc_score(self.labels, score_proba)
        eval_accuracy = accuracy_score(self.labels, (score_proba > 0.1).astype(int))
        eval_log_loss = log_loss(self.labels, score_proba)

        logger.info(
            "NDCG@5: {:.4f}\tNDCG@10: {:.4f}\tNDCG@20: {:.4f}\tRECALL@5: {:.4f}\tRECALL@10: {:.4f}\tRECALL@20: {:.4f}\tAUC: {:.4f}\tAcc: {:.4f}\tLog Loss: {:.4f}".format(
            eval_ndcg5, eval_ndcg10, eval_ndcg20,
            eval_recall5, eval_recall10, eval_recall20,
            eval_auc, eval_accuracy, eval_log_loss))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_ndcg5, eval_ndcg10, eval_ndcg20,
                    eval_recall5, eval_recall10, eval_recall20,
                    eval_auc, eval_accuracy, eval_log_loss])

        return eval_ndcg20
