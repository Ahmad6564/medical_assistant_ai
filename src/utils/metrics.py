"""
Evaluation metrics for Medical AI Assistant.
Provides metrics for NER, classification, ASR, and RAG tasks.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from collections import defaultdict
import jiwer


class NERMetrics:
    """Metrics for Named Entity Recognition."""
    
    @staticmethod
    def compute_entity_metrics(
        true_entities: List[List[Tuple[str, int, int]]],
        pred_entities: List[List[Tuple[str, int, int]]],
        entity_types: List[str]
    ) -> Dict[str, Any]:
        """
        Compute entity-level metrics (precision, recall, F1).
        
        Args:
            true_entities: Ground truth entities [(type, start, end), ...]
            pred_entities: Predicted entities [(type, start, end), ...]
            entity_types: List of entity types
            
        Returns:
            Dictionary of metrics per entity type and overall
        """
        metrics = {}
        
        for entity_type in entity_types + ["overall"]:
            tp = fp = fn = 0
            
            for true_ents, pred_ents in zip(true_entities, pred_entities):
                if entity_type == "overall":
                    true_set = set(true_ents)
                    pred_set = set(pred_ents)
                else:
                    true_set = set([e for e in true_ents if e[0] == entity_type])
                    pred_set = set([e for e in pred_ents if e[0] == entity_type])
                
                tp += len(true_set & pred_set)
                fp += len(pred_set - true_set)
                fn += len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
        
        return metrics
    
    @staticmethod
    def compute_token_metrics(
        y_true: List[List[str]],
        y_pred: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute token-level metrics.
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            
        Returns:
            Dictionary of token-level metrics
        """
        # Flatten sequences
        y_true_flat = [label for seq in y_true for label in seq]
        y_pred_flat = [label for seq in y_pred for label in seq]
        
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='weighted', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class ClassificationMetrics:
    """Metrics for clinical note classification."""
    
    @staticmethod
    def compute_multilabel_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compute metrics for multi-label classification.
        
        Args:
            y_true: True labels (shape: [n_samples, n_labels])
            y_pred: Predicted labels (shape: [n_samples, n_labels])
            label_names: Names of labels
            
        Returns:
            Dictionary of metrics per label and overall
        """
        metrics = {}
        
        # Overall metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        metrics["overall"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy_score(y_true, y_pred)
        }
        
        # Per-label metrics
        for i, label_name in enumerate(label_names):
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], average='binary', zero_division=0
            )
            
            metrics[label_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support)
            }
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrix for each label.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            label_names: Names of labels
            
        Returns:
            Dictionary of confusion matrices
        """
        matrices = {}
        
        for i, label_name in enumerate(label_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            matrices[label_name] = cm
        
        return matrices


class ASRMetrics:
    """Metrics for Automatic Speech Recognition."""
    
    @staticmethod
    def compute_wer(
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Compute Word Error Rate.
        
        Args:
            references: Reference transcriptions
            hypotheses: Predicted transcriptions
            
        Returns:
            Word Error Rate
        """
        return jiwer.wer(references, hypotheses)
    
    @staticmethod
    def compute_cer(
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Compute Character Error Rate.
        
        Args:
            references: Reference transcriptions
            hypotheses: Predicted transcriptions
            
        Returns:
            Character Error Rate
        """
        return jiwer.cer(references, hypotheses)
    
    @staticmethod
    def compute_medical_wer(
        references: List[str],
        hypotheses: List[str],
        medical_terms: List[str]
    ) -> Dict[str, float]:
        """
        Compute WER specifically for medical terms.
        
        Args:
            references: Reference transcriptions
            hypotheses: Predicted transcriptions
            medical_terms: List of medical terms to track
            
        Returns:
            Dictionary with overall and medical-specific WER
        """
        overall_wer = jiwer.wer(references, hypotheses)
        
        # Extract medical terms from references and hypotheses
        medical_refs = []
        medical_hyps = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            
            ref_medical = [w for w in ref_words if w in medical_terms]
            hyp_medical = [w for w in hyp_words if w in medical_terms]
            
            if ref_medical:
                medical_refs.append(" ".join(ref_medical))
                medical_hyps.append(" ".join(hyp_medical) if hyp_medical else "")
        
        medical_wer = jiwer.wer(medical_refs, medical_hyps) if medical_refs else 0.0
        
        return {
            "overall_wer": overall_wer,
            "medical_wer": medical_wer,
            "num_medical_terms": len(medical_refs)
        }


class RAGMetrics:
    """Metrics for Retrieval-Augmented Generation."""
    
    @staticmethod
    def compute_retrieval_metrics(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics (precision@k, recall@k, MRR).
        
        Args:
            retrieved_docs: Retrieved document IDs for each query
            relevant_docs: Relevant document IDs for each query
            k_values: Values of k for precision@k and recall@k
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            
            for retrieved, relevant in zip(retrieved_docs, relevant_docs):
                retrieved_at_k = set(retrieved[:k])
                relevant_set = set(relevant)
                
                if len(retrieved_at_k) > 0:
                    precision = len(retrieved_at_k & relevant_set) / len(retrieved_at_k)
                else:
                    precision = 0.0
                
                if len(relevant_set) > 0:
                    recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
                else:
                    recall = 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            metrics[f"precision@{k}"] = np.mean(precisions)
            metrics[f"recall@{k}"] = np.mean(recalls)
        
        # Mean Reciprocal Rank
        mrr_scores = []
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            relevant_set = set(relevant)
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    mrr_scores.append(1.0 / i)
                    break
            else:
                mrr_scores.append(0.0)
        
        metrics["mrr"] = np.mean(mrr_scores)
        
        return metrics
    
    @staticmethod
    def compute_answer_quality_metrics(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute answer quality metrics (placeholder for ROUGE, BLEU, etc.).
        
        Args:
            predictions: Generated answers
            references: Reference answers
            
        Returns:
            Dictionary of answer quality metrics
        """
        # Placeholder - would use libraries like rouge-score, sacrebleu
        return {
            "exact_match": np.mean([p == r for p, r in zip(predictions, references)]),
            "length_ratio": np.mean([len(p.split()) / len(r.split()) if len(r.split()) > 0 else 0 
                                    for p, r in zip(predictions, references)])
        }
