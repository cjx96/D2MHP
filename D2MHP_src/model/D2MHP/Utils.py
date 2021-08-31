import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    PAD = 0
    return seq.ne(PAD).type(torch.float)

def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased_original(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.intensity(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = F.softplus(temp_hid + model.alpha * temp_time, threshold=10)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral

def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, :-1]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (diff_time + 1).unsqueeze(2)

    temp_hid = model.intensity(data)[:, :-1, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = F.softplus(temp_hid + model.alpha * temp_time, threshold=10)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood_original(model, data, time, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.intensity(data)
    all_lambda = F.softplus(all_hid, threshold=10)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll

def log_likelihood(model, data, time, types, opt= {}):
    """ Log-likelihood of sequence. """
    non_pad_mask = pad_mask(types) # batch_size * seqs

    type_mask = torch.zeros([*types.size(), opt["eventnum"] + 1], device=data.device) # batch_size * seqs * num_types
    for i in range(opt["eventnum"] + 1):
        type_mask[:, :, i] = (types == i).bool().to(data.device)

    all_hid = model.intensity(data)
    all_lambda = F.softplus(all_hid, threshold=10)
    type_lambda = torch.sum(all_lambda[:,:-1,:] * type_mask[:,1:,:], dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask[:,:-1]) # batch_size * seqs
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def compute_integral_unbiased_distangle(model, mix_data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """
    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, :-1]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=mix_data.device)
    temp_time /= (diff_time + 1).unsqueeze(2)

    temp_hid = model.intensity(mix_data)[:, :-1, :]

    temp_hid = torch.sum(temp_hid, dim=2, keepdim=True)

    all_lambda = F.softplus(temp_hid + model.alpha * temp_time, threshold=10) # batch_size * seqs-1 * 1 batch_size * seqs-1 * 200
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples # batch_size * seqs-1

    unbiased_integral = all_lambda * diff_time # (batch_size * seqs-1) * (batch_size * seqs-1)
    return unbiased_integral

def log_likelihood_distangle(model, data, time, adapt, types, opt= {}):
    """ Log-likelihood of sequence. """
    types = types[:, :-1]
    time = time[:, :-1]

    non_pad_mask = pad_mask(types) # batch_size * seqs

    type_mask = torch.zeros([*types.size(), opt["eventnum"] + 1], device=data.device) # batch_size * seqs * num_types

    for i in range(opt["eventnum"] + 1):
        type_mask[:, :, i] = (types == i).bool().to(data.device)

    adapt = adapt.unsqueeze(-1) # batch_size * seqs * dis_k * 1
    mix_adapt = adapt.expand_as(data)
    mix_data = (data * mix_adapt).sum(-2)  # batch_size * seqs * dis_k * h_dim || batch_size * seqs * dis_k * h_dim
    # mix_data, batch_size * seqs * h_dim
    all_hid = model.intensity(mix_data) # batch_size * seqs * num_types
    all_lambda = F.softplus(all_hid, threshold=10) # batch_size * seqs * num_types
    type_lambda = torch.sum(all_lambda[:,:-1] * type_mask[:,1:], dim=-1)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask[:,:-1]) # batch_size * seqs
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased_distangle(model, mix_data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def compute_integral_unbiased_dvhp(model, mix_data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """
    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, :-1]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=mix_data.device)
    temp_time /= (diff_time + 1).unsqueeze(2)

    temp_hid = model.intensity(mix_data)[:, :-1, :] # batch_size * seqs-1 * num_types
    temp_hid = torch.sum(temp_hid, dim=2, keepdim=True)

    all_lambda = F.softplus(temp_hid + model.alpha * temp_time, threshold=10) # batch_size * seqs-1 * 1 batch_size * seqs-1 * 200
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples # batch_size * seqs-1

    unbiased_integral = all_lambda * diff_time # (batch_size * seqs-1) * (batch_size * seqs-1)
    return unbiased_integral


def HingeLoss(pos, neg, gap = -1, opt = {}):
    if gap<0:
        gamma = torch.tensor(0.1)
    else:
        gamma = torch.tensor(gap)
    if opt["cuda"]:
        gamma = gamma.cuda()
    return F.relu(gamma - pos + neg).mean()


def get_hawkes_mask(alpha_mask):
    return (alpha_mask>0).type(torch.float)

def sample_hawkes_mask(alpha_mask, sample_mask, num_samples):

    shift_time = torch.rand([alpha_mask.size()[0], num_samples], device = alpha_mask.device) # batch_size * num_samples

    shift_time = shift_time.unsqueeze(-1) # batch_size * num_samples * 1
    shift_time = shift_time.repeat(1,1,alpha_mask.size()[2]) # batch_size * num_samples * event_num

    shift_time = shift_time.unsqueeze(2)  # batch_size * num_samples * 1 * event
    shift_time = shift_time.repeat(1, 1, alpha_mask.size()[1],1)  # batch_size * num_samples * dis_k * event_num

    sample_mask = sample_mask.unsqueeze(1)
    sample_mask = sample_mask.repeat(1,num_samples,1,1)
    alpha_mask = alpha_mask.unsqueeze(1)
    alpha_mask = alpha_mask.repeat(1, num_samples, 1, 1)
    return alpha_mask + shift_time * sample_mask # batch_size * num_samples * dis_k * event_num

def log_likelihood_dvhp(pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, adapt, opt):
    """ Log-likelihood of sequence. """

    (pos_event_alpha, pos_event_delta, pos_alpha_mask) = pos_alpha
    (neg_event_alpha, neg_event_delta, neg_alpha_mask) = neg_alpha

    # dis_mu batch_size * seqs * dis_k
    # dis_alpha batch_size * seqs * dis_k

    # pos_event_alpha batch_size * dis_k * num_event
    # pos_alpha_mask batch_size * dis_k * num_event
    num_samples = 100

    sample_mask = get_hawkes_mask(pos_alpha_mask) # batch_size * dis_k * event_num
    pos_alpha_time_mask = sample_hawkes_mask(pos_alpha_mask, sample_mask, num_samples) # batch_size * num_samples * dis_k * event_num
    # import pdb
    # pdb.set_trace()
    pos_event_alpha = pos_event_alpha.unsqueeze(1).repeat(1,num_samples,1,1) # batch_size * num_samples * dis_k * event_num
    pos_event_delta = pos_event_delta.unsqueeze(1).repeat(1, num_samples, 1, 1) # batch_size * num_samples * dis_k * event_num
    pos_dis_alpha = (pos_event_alpha * torch.exp(-pos_event_delta * pos_alpha_time_mask) * pos_alpha_mask).sum(dim=-1) # batch_size * num_samples * dis_k

    pos_alpha_mask = get_hawkes_mask(pos_alpha_time_mask)
    neg_event_alpha = neg_event_alpha.unsqueeze(1).repeat(1, num_samples, 1, 1)
    neg_event_delta = neg_event_delta.unsqueeze(1).repeat(1, num_samples, 1, 1)
    neg_dis_alpha = (neg_event_alpha * torch.exp(-neg_event_delta * pos_alpha_time_mask) * pos_alpha_mask).sum(dim=-1) # batch_size * num_samples * dis_k

    adapt = adapt.unsqueeze(1).repeat(1, num_samples ,1) # batch_size * num_samples * dis_k
    pos_dis_mu = pos_dis_mu.unsqueeze(1).repeat(1, num_samples, 1) # batch_size * num_samples * dis_k
    pos_mix_data = (pos_dis_alpha * adapt + pos_dis_mu).sum(-1)  # batch_size * num_samples
    pos_all_lambda = F.softplus(pos_mix_data, threshold=10) # batch_size * num_samples
    # pos_all_lambda = F.softplus(pos_mix_data)  # batch_size * seqs * num_types

    neg_dis_mu = neg_dis_mu.unsqueeze(1).repeat(1, num_samples, 1) # batch_size * num_samples * dis_k
    neg_mix_data = (neg_dis_alpha * adapt + neg_dis_mu).sum(-1)  # batch_size * num_samples
    neg_all_lambda = F.softplus(neg_mix_data, threshold=10)  # # batch_size * num_samples
    # neg_all_lambda = F.softplus(neg_mix_data)  # batch_size * seqs * num_types

    event_ll = HingeLoss(pos_all_lambda, neg_all_lambda, 0.3, opt)
    return event_ll


def sample_hawkes_mask_adapt(alpha_mask, sample_mask, num_samples):

    shift_time = torch.rand([alpha_mask.size()[0], num_samples], device = alpha_mask.device) # batch_size * num_samples

    shift_time = shift_time.unsqueeze(-1) # batch_size * num_samples * 1
    tmp_shift_time = shift_time
    shift_time = shift_time.repeat(1,1,alpha_mask.size()[2]) # batch_size * num_samples * event_num
    tmp_shift_time = tmp_shift_time.repeat(1, 1, alpha_mask.size()[1]) # # batch_size * num_samples * dis_k
    shift_time = shift_time.unsqueeze(2)  # batch_size * num_samples * 1 * event
    shift_time = shift_time.repeat(1, 1, alpha_mask.size()[1],1)  # batch_size * num_samples * dis_k * event_num

    sample_mask = sample_mask.unsqueeze(1)
    sample_mask = sample_mask.repeat(1,num_samples,1,1)
    alpha_mask = alpha_mask.unsqueeze(1)
    alpha_mask = alpha_mask.repeat(1, num_samples, 1, 1) # batch_size * num_samples * dis_k * event_num
    return alpha_mask + shift_time * sample_mask, tmp_shift_time  # batch_size * num_samples * dis_k * event_num

def log_likelihood_dvhp_adapt(pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, adapt, adapt_bar, opt):
    """ Log-likelihood of sequence. """

    (pos_event_alpha, pos_event_delta, pos_alpha_time_mask) = pos_alpha
    (neg_event_alpha, neg_event_delta, neg_alpha_mask) = neg_alpha

    # dis_mu batch_size * seqs * dis_k
    # dis_alpha batch_size * seqs * dis_k

    # pos_event_alpha batch_size * dis_k * num_event
    # pos_alpha_mask batch_size * dis_k * num_event
    num_samples = 100

    sample_mask = get_hawkes_mask(pos_alpha_time_mask) # batch_size * dis_k * event_num
    pos_alpha_time_mask, tmp_shift_time = sample_hawkes_mask_adapt(pos_alpha_time_mask, sample_mask, num_samples) # batch_size * num_samples * dis_k * event_num
    # import pdb
    # pdb.set_trace()
    pos_alpha_mask = get_hawkes_mask(pos_alpha_time_mask)

    pos_event_alpha = pos_event_alpha.unsqueeze(1).repeat(1,num_samples,1,1) # batch_size * num_samples * dis_k * event_num
    pos_event_delta = pos_event_delta.unsqueeze(1).repeat(1, num_samples, 1, 1) # batch_size * num_samples * dis_k * event_num
    pos_dis_alpha = (pos_event_alpha * torch.exp(-pos_event_delta * pos_alpha_time_mask) * pos_alpha_mask).sum(dim=-1) # batch_size * num_samples * dis_k


    neg_event_alpha = neg_event_alpha.unsqueeze(1).repeat(1, num_samples, 1, 1)
    neg_event_delta = neg_event_delta.unsqueeze(1).repeat(1, num_samples, 1, 1)
    neg_dis_alpha = (neg_event_alpha * torch.exp(-neg_event_delta * pos_alpha_time_mask) * pos_alpha_mask).sum(dim=-1) # batch_size * num_samples * dis_k

    adapt = adapt.unsqueeze(1).repeat(1, num_samples ,1) # batch_size * num_samples * dis_k
    adapt_bar = adapt_bar.unsqueeze(1).repeat(1, num_samples, 1)  # batch_size * num_samples * dis_k
    pos_dis_mu = pos_dis_mu.unsqueeze(1).repeat(1, num_samples, 1) # batch_size * num_samples * dis_k
    pos_mix_data = (pos_dis_alpha * (adapt_bar + (adapt-adapt_bar) * torch.exp(-tmp_shift_time)) + pos_dis_mu).sum(-1)  # batch_size * num_samples
    pos_all_lambda = F.softplus(pos_mix_data, threshold=10) # batch_size * num_samples
    # pos_all_lambda = F.softplus(pos_mix_data)  # batch_size * seqs * num_types

    neg_dis_mu = neg_dis_mu.unsqueeze(1).repeat(1, num_samples, 1) # batch_size * num_samples * dis_k
    neg_mix_data = (neg_dis_alpha * (adapt_bar + (adapt-adapt_bar) * torch.exp(-tmp_shift_time)) + neg_dis_mu).sum(-1)  # batch_size * num_samples
    neg_all_lambda = F.softplus(neg_mix_data, threshold=10)  # # batch_size * num_samples
    # neg_all_lambda = F.softplus(neg_mix_data)  # batch_size * seqs * num_types

    event_ll = HingeLoss(pos_all_lambda, neg_all_lambda, 0.3, opt)
    return event_ll

def log_likelihood_dvhp_adapt_nll(pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, adapt, adapt_bar, opt):
    """ Log-likelihood of sequence. """

    (pos_event_alpha, pos_event_delta, pos_alpha_time_mask) = pos_alpha
    (neg_event_alpha, neg_event_delta, neg_alpha_mask) = neg_alpha

    # dis_mu batch_size * seqs * dis_k
    # dis_alpha batch_size * seqs * dis_k

    # pos_event_alpha batch_size * dis_k * num_event
    # pos_alpha_mask batch_size * dis_k * num_event
    num_samples = 100

    sample_mask = get_hawkes_mask(pos_alpha_time_mask) # batch_size * dis_k * event_num
    pos_alpha_time_mask, tmp_shift_time = sample_hawkes_mask_adapt(pos_alpha_time_mask, sample_mask, num_samples) # batch_size * num_samples * dis_k * event_num
    # import pdb
    # pdb.set_trace()
    pos_alpha_mask = get_hawkes_mask(pos_alpha_time_mask)
    pos_event_alpha = pos_event_alpha.unsqueeze(1).repeat(1,num_samples,1,1) # batch_size * num_samples * dis_k * event_num
    pos_event_delta = pos_event_delta.unsqueeze(1).repeat(1, num_samples, 1, 1) # batch_size * num_samples * dis_k * event_num
    pos_dis_alpha = (pos_event_alpha * torch.exp(-pos_event_delta * pos_alpha_time_mask) * pos_alpha_mask).sum(dim=-1) # batch_size * num_samples * dis_k

    neg_event_alpha = neg_event_alpha.unsqueeze(1).repeat(1, num_samples, 1, 1)
    neg_event_delta = neg_event_delta.unsqueeze(1).repeat(1, num_samples, 1, 1)
    neg_dis_alpha = (neg_event_alpha * torch.exp(-neg_event_delta * pos_alpha_time_mask) * pos_alpha_mask).sum(dim=-1) # batch_size * num_samples * dis_k

    adapt = adapt.unsqueeze(1).repeat(1, num_samples ,1) # batch_size * num_samples * dis_k
    adapt_bar = adapt_bar.unsqueeze(1).repeat(1, num_samples, 1)  # batch_size * num_samples * dis_k
    pos_dis_mu = pos_dis_mu.unsqueeze(1).repeat(1, num_samples, 1) # batch_size * num_samples * dis_k
    pos_mix_data = (pos_dis_alpha * (adapt_bar + (adapt-adapt_bar) * torch.exp(-tmp_shift_time)) + pos_dis_mu).sum(-1)  # batch_size * num_samples
    pos_all_lambda = torch.clamp(F.softplus(pos_mix_data, threshold=2) / adapt.size()[2] / num_samples, min=0, max=2) # batch_size * num_samples
    # pos_all_lambda = F.sigmoid(pos_mix_data)  # batch_size * num_samples
    pos_all_lambda_ll = pos_all_lambda.mean(-1)

    neg_dis_mu = neg_dis_mu.unsqueeze(1).repeat(1, num_samples, 1) # batch_size * num_samples * dis_k
    neg_mix_data = (neg_dis_alpha * (adapt_bar + (adapt-adapt_bar) * torch.exp(-tmp_shift_time)) + neg_dis_mu).sum(-1)  # batch_size * num_samples
    neg_all_lambda = torch.clamp(F.softplus(neg_mix_data, threshold=2) / adapt.size()[2] / num_samples, min=0, max=2) # # batch_size * num_samples
    # neg_all_lambda = F.sigmoid(neg_mix_data)  # batch_size * num_samples

    event_ll = torch.sum(torch.clamp(torch.log(pos_all_lambda_ll), min=-2, max=50))
    # event_nll = torch.log(torch.sum(pos_all_lambda))

    # batch_size * (num_samples-1), batch_size * (num_samples-1)
    no_event_ll = torch.sum((pos_all_lambda + 2*neg_all_lambda)[:,1:] * (tmp_shift_time[:, :, 0][:,1:] - tmp_shift_time[:, :, 0][:,:-1]))
    # no_event_nll = torch.sum((neg_all_lambda * tmp_shift_time[:, :, 0]).mean())

    ll = event_ll - no_event_ll
    return ll

def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    time_mask = torch.tensor(event_time[:, :-1] > 1).float()
    # print("time_mask:")
    # print(time_mask)

    # print(event_time)
    true = (event_time[:, 1:] - event_time[:, :-1]).float()

    # print("true:")
    # print(true)

    prediction = prediction[:, :-1]

    # print(true.size())
    # print(prediction.size())

    # event time gap prediction
    diff = (prediction - true) * time_mask

    se = torch.sum(diff * diff).mean()
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
