def epoch_update_gamma(y_true, y_pred, epoch=-1, delta=1.0):
    """
    Calculate gamma from last epoch's targets and predictions.
    Adapted from C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."

    y_true: Targets with value in {0, 1}.
    y_pred: Predictions with value in [0, 1].
    """

    # Hyperparameters
    DELTA = delta
    SUB_SAMPLE_SIZE = 2000.0

    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]

    # subsample the training set for performance
    cap_pos = pos.shape[0]
    cap_neg = neg.shape[0]
    pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE / cap_pos]
    neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE / cap_neg]

    # number of positive and negative samples
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    pos_expand = pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)
    diff = neg_expand - pos_expand
    ln_All = diff.shape[0]

    # flip the pos and neg for loss computation
    Lp = diff[diff > 0]
    ln_Lp = Lp.shape[0] - 1
    diff_neg = -1.0 * diff[diff < 0]
    diff_neg = diff_neg.sort()[0]
    ln_neg = diff_neg.shape[0] - 1
    ln_neg = max([ln_neg, 0])
    left_wing = int(ln_Lp * DELTA)
    left_wing = max([0, left_wing])
    left_wing = min([ln_neg, left_wing])
    default_gamma = torch.tensor(0.2, dtype=torch.float).cuda()
    if diff_neg.shape[0] > 0:
        gamma = diff_neg[left_wing]
    else:
        gamma = default_gamma  # default=torch.tensor(0.2, dtype=torch.float).cuda()
    L1 = diff[diff > -1.0 * gamma]
    ln_L1 = L1.shape[0]
    if epoch > -1:
        return gamma
    else:
        return default_gamma


def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    Calculate the ROC loss as defined in this paper: https://www.aaai.org/Library/ICML/2003/icml03-110.php.
    Adapted from C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/roc-star

    _y_true: Targets with value in {0, 1}.
    y_pred: Predictions with value in [0, 1]/
    gamma  : Gamma as obtained from last epoch.
    _epoch_true: Targets from last epoch.
    epoch_pred : Predicions from last epoch.
    """
    # convert labels to boolean
    y_true = (_y_true >= 0.50)
    epoch_true = (_epoch_true >= 0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred) * 1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000  # Max number of positive training samples
    max_neg = 1000  # Max number of negative training samples
    cap_pos = epoch_pos.shape[0]
    cap_neg = epoch_neg.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos / cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg / cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements against (subsampled) negative elements
    if ln_pos > 0:
        pos_expand = pos.view(-1, 1).expand(-1, epoch_neg.shape[0]).reshape(-1).float()
        neg_expand = epoch_neg.repeat(ln_pos).float()

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2 > 0]
        m2 = l2 * l2
        len2 = l2.shape[0]
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()
        len2 = 0

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg > 0:
        pos_expand = epoch_pos.view(-1, 1).expand(-1, ln_neg).reshape(-1).float()
        neg_expand = neg.repeat(epoch_pos.shape[0]).float()

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3 > 0]
        m3 = l3 * l3
        len3 = l3.shape[0]
    else:
        m3 = torch.tensor([0], dtype=torch.double).cuda()
        len3 = 0

    if (torch.sum(m2) + torch.sum(m3)) != 0:
        res2 = torch.sum(m2) / max_pos + torch.sum(m3) / max_neg
    else:
        res2 = torch.sum(m2) + torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2