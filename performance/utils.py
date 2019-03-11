import torch
import torch.nn.functional as F
import torch.nn as nn


def evaluation(x, guide, T=10):
    """
    sampled from variational distribution T times, and evaluate the output based on the evaluation
    assume guide has format guide(x,y)
    :param x: input
    :param guide: pyro guide
    :param T: number of times sample from variational distribution
    :return: logits
    """

    sampled_models = [guide(None, None) for _ in range(T)]
    yhats = [model(x).data for model in sampled_models]
    yhats = torch.stack(yhats, dim=1)  # [batch * T * dim]
    return yhats


def softmax_logits(loader, guide, shape, T=10, use_cuda=False):
    """
    return softmax(logits) given input x of the network, each x with T weights sampled from variational distribution
    :param loader: pytorch data lodaer
    :param guide: pyro guide
    :param shape: list, input data shape, reshaped by torch.tensor.view
    :param T: number of weights sampled from the variational distribution for each input x, default: 10
    :param use_cuda: if cuda available, default False
    :return: torch.tensor(softmax(logits)), torch.tensor(y)
    """

    prob_list = []
    label_list = []
    for j, data in enumerate(loader):
        if use_cuda:
            x = data[0].view(shape).cuda()
        else:
            x = data[0].view(shape)
        y = data[1]

        logits = evaluation(x, guide, T)  # [batch * T * dim]
        mean_prob = torch.mean(F.softmax(logits, dim=2), 1)  # [batch * dim]

        if use_cuda:
            prob_list.append(mean_prob.cpu())
            label_list.append(y.cpu())
        else:
            prob_list.append(mean_prob)
            label_list.append(y)

    return torch.cat(prob_list, dim=0), torch.cat(label_list, dim=0)


def uncertainty(loader, guide, shape, T=10, use_cuda=False):
    """
    estimatet the epistemic uncertainty and aleatoric uncertainty
    :param loader: pytorch data lodaer
    :param guide: pyro guide
    :param shape: input data reshape
    :param threshold: confidence threshold
    :param T: number of times sample from variational distribution
    :param use_cuda: if cuda available, default False
    :return: alea, epis, prob | torch.tensor
    """
    def uncertainty_cal(x, guide, T, use_cuda=False):
        logits = evaluation(x, guide, T)  # [batch * T * dim]
        p_hat = F.softmax(logits, dim=2)  # [batch * T * dim]

        alea = torch.mean(p_hat * (1 - p_hat), dim=1)  # [batch * dim]
        epis = torch.sub(torch.mean(p_hat ** 2, dim=1), torch.mean(p_hat, dim=1) ** 2)
        mean = torch.mean(p_hat, dim=1)

        if use_cuda:
            return alea.cpu(), epis.cpu(), mean.cpu()
        else:
            return alea, epis, mean

    alea_list = []
    epis_list = []
    prob_list = []
    label_list = []

    for j, data in enumerate(loader):
        if use_cuda:
            x = data[0].view(shape).cuda()
        else:
            x = data[0].view(shape)
        y = data[1]


        alea, epis, prob = uncertainty_cal(x, guide, T, use_cuda)

        alea_list.append(alea)
        epis_list.append(epis)
        prob_list.append(prob)
        label_list.append(y)

    alea_list, epis_list, prob_list, label_list = torch.cat(alea, dim=0), torch.cat(epis_list, dim=0), \
                                                  torch.cat(prob_list, dim=0), torch.cat(label_list, dim=0)  # [batch * dim]

    return alea_list, epis_list, prob_list, label_list


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ReliabilityDiagram(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ReliabilityDiagram, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        x = []
        y = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                x.append(avg_confidence_in_bin)
                y.append(accuracy_in_bin)
        return torch.stack(x, dim=0).view(-1).cpu().numpy(), torch.stack(y, dim=0).view(-1).cpu().numpy()