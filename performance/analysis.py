import torch
from performance.utils import softmax_logits, uncertainty, ECELoss, ReliabilityDiagram
import numpy as np
import pandas as pd


def accuracy_score(loader, guide, shape, threshold, T=10, use_cuda=False):
    """ return accuracy, number of samples been rejected, the reject ratio"""
    score, label = softmax_logits(loader, guide, shape, T, use_cuda)
    confidence, predictions = torch.max(score, dim=1)

    confidence, predictions = confidence.numpy(), predictions.numpy()
    label = label.numpy()

    # get index of prediction with confidence higher than threshold
    idx = [idx for idx in range(len(confidence)) if confidence[idx] >= threshold]

    # calculate accuracy
    correct = (predictions[idx] == label[idx]).sum().item()

    # results
    acc = correct/len(idx)
    skipped = len(confidence) - len(idx)
    rejection_ratio = skipped/len(confidence)
    return acc, skipped, rejection_ratio


def uncertainty_threshold(loader, guide, shape, threshold, T=10, use_cuda=False):
    """ calculate aleatoric uncertainty and epistemic uncertainty"""
    alea, epis, prob, label = uncertainty(loader, guide, shape, T, use_cuda)
    confidence, predictions = torch.max(prob, dim=1)

    confidence, predictions, alea, epis, label = confidence.numpy(), predictions.numpy(), alea.numpy(), epis.numpy(), \
                                                 label.numpy()

    idx = [idx for idx in range(len(confidence)) if confidence[idx] >= threshold]
    correct = [each[1] for each in zip(predictions[idx]==label[idx], idx) if each[0] is True]
    wrong = [each[1] for each in zip(predictions[idx]==label[idx], idx) if each[0] is False]

    mean_alea_total = np.choose(predictions, alea.T)[idx].sum().item()/len(idx)
    mean_epis_total = np.choose(predictions, epis.T)[idx].sum().item()/len(idx)

    mean_alea_correct = np.choose(predictions, alea.T)[correct].sum().item()/len(correct)
    mean_epis_correct = np.choose(predictions, epis.T)[correct].sum().item()/len(correct)

    mean_alea_wrong = np.choose(predictions, alea.T)[wrong].sum().item()/len(wrong)
    mean_epis_wrong = np.choose(predictions, alea.T)[wrong].sum().item()/len(wrong)

    return mean_alea_total, mean_epis_total, mean_alea_correct, mean_epis_correct, mean_alea_wrong, mean_epis_wrong


def ece_loss_cal(loader, guide, shape, T=10, n_bins=15, use_cuda=False):
    """ calculate expected calibration error"""
    score, label = softmax_logits(loader, guide, shape, T, use_cuda)
    ece = ECELoss(n_bins=n_bins)
    return ece.forward(score, label)


def reliability_diagram(loader, guide, shape, T=10, n_bins=15, use_cuda=False):
    score, label = softmax_logits(loader, guide, shape, T, use_cuda)
    rd = ReliabilityDiagram(n_bins=n_bins)
    confidence, accuracy = rd(score, label)
    return confidence, accuracy


def empirical_cdf_prob(loader, guide, shape, T=10, use_cuda=False):
    """return x (probability), y (cdf of probability)"""
    score, label = softmax_logits(loader, guide, shape, T, use_cuda)
    confidence, predictions = torch.max(score, dim=1)

    prob_cnt = pd.Series(confidence).value_counts().sort_index()
    cumulative = np.cumsum(prob_cnt.values)
    
    return prob_cnt.index, cumulative