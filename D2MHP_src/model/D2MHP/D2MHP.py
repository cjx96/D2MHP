import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class D2MHP(nn.Module):
    def __init__(self, opt):
        super(D2MHP, self).__init__()

        self.opt = opt

        x_dim = opt["x_dim"]
        h_dim = opt["h_dim"]
        z_dim = opt["z_dim"]
        n_layers = opt["n_layers"]

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dis_k = opt["dis_k"]
        self.identity_dim = opt["identity_dim"]
        self.dis_dim = opt["identity_dim"] * opt["dis_k"]
        self.n_layers = n_layers

        self.f_k = nn.Bilinear(self.h_dim, self.h_dim, 1)

        self.event_emb = nn.Embedding(opt["eventnum"] + 1, self.x_dim, padding_idx=0)

        nn.init.normal_(self.event_emb.weight, 0, 1)

        # our intensity calculate
        self.mu_encoder = []
        for i in range(opt["dis_k"]):
            self.mu_encoder.append(nn.Linear(h_dim + x_dim, 1))
        self.mu_encoder = nn.ModuleList(self.mu_encoder)

        self.delta = nn.Embedding(opt["eventnum"] + 1, (opt["eventnum"] + 1) * opt["dis_k"], padding_idx=0)

        # feature-extracting transformations
        self.phi_x = nn.Sequential()
            # nn.Linear(x_dim, h_dim),
            # nn.ReLU())

        self.phi_z_c = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        self.phi_z_t = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # piror
        self.prior_enc = nn.Sequential(
            nn.Linear(self.identity_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim))

        # post
        self.post_enc = nn.Sequential(
            nn.Linear(self.identity_dim, h_dim),
            nn.ReLU())
        self.post_mean = nn.Linear(h_dim, z_dim)
        self.post_std = nn.Sequential(
        nn.Linear(h_dim, z_dim))

        self.z_c_enc = nn.Sequential(
            nn.Linear(self.identity_dim, self.h_dim),
            nn.ReLU())
        self.z_c_mean = nn.Linear(self.h_dim, z_dim)
        self.z_c_std = nn.Sequential(
            nn.Linear(self.h_dim, z_dim))

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim + z_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim)
        )

        # adapt
        self.adapt = nn.Sequential(
            nn.Linear(h_dim + h_dim, 1),
            nn.ReLU())

        self.adapt_bar = nn.Sequential(
            nn.Linear(h_dim + h_dim, 1),
            nn.ReLU())

        self.prior_rnn = nn.GRU(h_dim, self.x_dim, n_layers, batch_first=True)
        self.post_rnn = nn.GRU(h_dim, self.x_dim, n_layers, batch_first=True)
        self.prior_fc = nn.Linear(self.x_dim, self.dis_dim)
        self.post_fc = nn.Linear(self.x_dim, self.dis_dim)
        self.z_c_x_to_dis = nn.Linear(self.x_dim, self.dis_dim)
        self.PE = PositionalEmbedding(opt)
        self.biasedPE = BiasedPositionalEmbedding(opt)

        # variational prediction
        self.time_predicter = nn.Linear(2*h_dim, 1)
        self.z_c_predicter = nn.Linear(self.dis_k * h_dim, self.h_dim)
        self.z_t_predicter = nn.Linear(self.dis_k * h_dim, self.h_dim)
        self.event_predicter = nn.Linear(3*h_dim, self.x_dim)
        # basic hawkes process parameters
        self.alpha_adapt_gamma = nn.Linear(self.h_dim, self.opt["eventnum"] + 1)
        self.alpha_adapt_beta = nn.Linear(self.h_dim, self.opt["eventnum"] + 1)
        self.delta_adapt_gamma = nn.Linear(self.h_dim, self.opt["eventnum"] + 1)
        self.delta_adapt_beta = nn.Linear(self.h_dim, self.opt["eventnum"] + 1)
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _reparameterized_sample_orignal(self, mean, logvariance):
        """using std to sample"""
        eps = torch.FloatTensor(logvariance.size()).normal_().cuda()
        eps = Variable(eps)
        std = torch.exp(0.5 * logvariance)
        return eps.mul(std).add_(mean)

    def _reparameterized_sample(self, mu, sigma):
        """using std to sample"""
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        eps = torch.FloatTensor(sigma.size()).normal_().cuda()
        eps = Variable(eps)
        return eps.mul(sigma).add_(mu)

    def _kld_gauss(self, mu_1, sigma_1, mu_2, sigma_2):
        """Using std to compute KLD"""
        # sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(sigma_1))
        # sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(sigma_2))
        sigma_1 = 0.1 + 0.9 * F.softplus(sigma_1)
        sigma_2 = 0.1 + 0.9 * F.softplus(sigma_2)
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def dump_index_select(self, memory, index):
        tmp = list(index.size()) + [self.dis_k, -1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def seq2feats(self, event_input, time_input, pos_seqs, neg_seqs, alpha_mask):

        x = self.event_emb(event_input)
        prior_h = Variable(torch.zeros(self.n_layers, x.size(0), self.x_dim).cuda())  # [layers, batch_size, x_dim]
        post_h = Variable(torch.zeros(self.n_layers, x.size(0), self.x_dim).cuda())  # [layers, batch_size, x_dim]
        kld_loss = 0

        seqs_len = event_input.size()[1]
        index = torch.arange(0, seqs_len, 1)
        if self.opt["cuda"]:
            index = index.cuda()
        position_embedding = self.PE(index)
        position_embedding = position_embedding.unsqueeze(0)
        position_embedding = position_embedding.repeat(event_input.size()[0], 1, 1)
        biased_position_embedding = self.biasedPE(time_input)

        phi_x = self.phi_x(x) + biased_position_embedding # batch_size * seqs * x_dim

        z_c_x = torch.mean(phi_x, -2) # batch_size * x_dim
        z_c_dis = self.z_c_x_to_dis(z_c_x) # batch_size * dis_dim
        post_h, last_post_h = self.post_rnn(phi_x, post_h) # 128 * 200 * x_dim
        prior_h, last_prior_h = self.prior_rnn(phi_x, prior_h) # 128 * 200 * x_dim
        post_h_dis = self.post_fc(post_h) # 128 * 200 * dis_dim
        prior_h_dis = self.prior_fc(prior_h) # 128 * 200 * dis_dim

        z_c_dis = z_c_dis.view(z_c_dis.size()[0], self.dis_k, -1)
        z_c_h = self.z_c_enc(z_c_dis) # 128 * dis_k * h_dim
        z_c_mean = self.z_c_mean(z_c_h)  # batch_size * dis_k * z_dim
        z_c_std = self.z_c_std(z_c_h)  # batch_size * dis_k * z_dim
        z_c = self._reparameterized_sample(z_c_mean, z_c_std)  # batch_size * dis_k * z_dim
        z_c = self.phi_z_c(z_c)  # batch_size * dis_k * h_dim

        z_c_h_true = z_c_h
        z_c_h_false_index = torch.randperm(z_c_h.size()[0], device= z_c_h.device)
        z_c_h_false = self.dump_index_select(z_c_h, z_c_h_false_index)
        z_c_ture = self.f_k(z_c, z_c_h_true).squeeze(-1)
        z_c_false = self.f_k(z_c, z_c_h_false).squeeze(-1)
        pos_labels, neg_labels = torch.ones_like(z_c_ture,device=z_c_ture.device), torch.zeros_like(
            z_c_false, device=z_c_false.device)
        mi_loss = self.BCEWithLogitsLoss(z_c_ture, pos_labels)
        mi_loss += self.BCEWithLogitsLoss(z_c_false, neg_labels)

        kld_loss +=  self._kld_gauss(z_c_mean, z_c_std, torch.zeros_like(z_c_mean), torch.ones_like(z_c_std))

        post_h_dis = post_h_dis.view(post_h_dis.size()[0], post_h.size()[1], self.dis_k, -1)
        post_h_di_to_zt = self.post_enc(post_h_dis) # batch_size * seqs * dis_k * z_dim
        post_mean = self.post_mean(post_h_di_to_zt) # batch_size * seqs * dis_k * z_dim
        post_std = self.post_std(post_h_di_to_zt) # batch_size * seqs * dis_k * z_dim

        prior_h_dis = prior_h_dis.view(prior_h_dis.size()[0], prior_h.size()[1], self.dis_k, -1) # batch_size * seqs * dis_k * identity_dim
        prior_h_dis_to_zt = self.prior_enc(prior_h_dis)
        prior_mean = self.prior_mean(prior_h_dis_to_zt)
        prior_std = self.prior_std(prior_h_dis_to_zt)

        kld_loss += self._kld_gauss(post_mean[:,1:], post_std[:,1:], prior_mean[:,:-1], prior_std[:,:-1])

        z_t = self._reparameterized_sample(post_mean, post_std) # batch_size * seqs * dis_k * z_dim
        phi_z_t = self.phi_z_t(z_t) # batch_size * seqs * dis_k * h_dim

        z = torch.cat((phi_z_t[:,-1,:,:], z_c), dim = -1) # batch_size * dis_k * (2*h_dim)

        z_adapt = self.adapt(z).squeeze(-1) #
        z_adapt_bar = self.adapt_bar(z).squeeze(-1)  #

        z_adapt = F.softmax(z_adapt, dim = -1) #
        z_adapt_bar = F.softmax(z_adapt_bar, dim=-1)  #

        p_time_dis = self.time_predicter(z).squeeze(-1) # batch_size * dis_k
        p_time_dis = p_time_dis * z_adapt_bar # batch_size * dis_k
        p_time = torch.sum(p_time_dis, dim = -1) # batch_size
        # dec = torch.cat((z, prior_h_dis_to_zt[:,:-1]),dim = -1) # batch_size * seqs * (2*h_dim + z_dim)
        # dec = self.dec(dec) # batch_size * seqs * x_dim

        v_z_c = self.z_c_predicter(z_c.view(z_c.size()[0],-1)).unsqueeze(1).repeat(1,prior_h.size()[1],1)
        v_z_t = self.z_t_predicter(z_t.view(z_t.size()[0], z_t.size()[1], -1))
        enc_output = self.event_predicter(torch.cat((prior_h[:,:-1], v_z_c[:,:-1], v_z_t[:,:-1]),-1)) # batch_size * (seq_len-1) * x_dim

        pos_dis_mu = torch.zeros((z_c.size()[0], z_c.size()[1])) # batch_size * dis_k
        if self.opt["cuda"]:
            pos_dis_mu = pos_dis_mu.cuda()
        for b_id in range(len(pos_seqs)):
            # pos_dis_mu[b_id] = dis_mu[b_id, : , pos_seqs[b_id,-2]] # batch_size * dis_k
            pos_event = self.event_emb.weight[pos_seqs[b_id,-2]] # x_dim
            pos_event = pos_event.unsqueeze(0).repeat(z_c.size()[1],1) # dis_k, x_dim
            pos_event = torch.cat((z_c[b_id],pos_event),dim = -1) # dis_k, (h_dim + x_dim)
            for i in range(self.opt["dis_k"]):
                pos_dis_mu[b_id, i] = self.mu_encoder[i](pos_event[i]).unsqueeze(-1)  # batch_size * dis_k


        neg_dis_mu = torch.zeros((z_c.size()[0], z_c.size()[1]))  # batch_size * dis_k
        if self.opt["cuda"]:
            neg_dis_mu = neg_dis_mu.cuda()
        for b_id in range(len(neg_seqs)):
            neg_event = self.event_emb.weight[neg_seqs[b_id, -2]]  # x_dim
            neg_event = neg_event.unsqueeze(0).repeat(z_c.size()[1], 1)  # dis_k, x_dim
            neg_event = torch.cat((z_c[b_id], neg_event), dim=-1)  # dis_k, (h_dim + x_dim)
            for i in range(self.opt["dis_k"]):
                neg_dis_mu[b_id, i] = self.mu_encoder[i](neg_event[i]).unsqueeze(-1)  # batch_size * dis_k

        alpha_mask = alpha_mask.unsqueeze(1).repeat(1,z_c.size()[1],1) # batch_size * dis_k * num_event

        alpha_adapt_gamma = self.alpha_adapt_gamma(phi_z_t[:,-1]) # batch_size * dis_k * num_event
        alpha_adapt_gamma = torch.tanh(alpha_adapt_gamma)
        alpha_adapt_beta = self.alpha_adapt_beta(phi_z_t[:,-1]) # batch_size  * dis_k * num_event
        alpha_adapt_beta = torch.tanh(alpha_adapt_beta)
        delta_adapt_gamma = self.delta_adapt_gamma(phi_z_t[:, -1])  # batch_size * dis_k * num_event
        delta_adapt_gamma = torch.tanh(delta_adapt_gamma)
        delta_adapt_beta = self.delta_adapt_beta(phi_z_t[:, -1])  # batch_size  * dis_k * num_event
        delta_adapt_beta = torch.tanh(delta_adapt_beta)

        pos_event_alpha = self.event_emb.weight[pos_seqs[:, -2]].unsqueeze(1) # batch_size * 1 * x_dim
        pos_event_all_alpha_T = self.event_emb.weight[:].transpose(0,1).unsqueeze(0) # 1 * x_dim * num_event
        pos_event_alpha = torch.matmul(pos_event_alpha, pos_event_all_alpha_T) # batch_size * 1 * num_event
        pos_event_alpha = pos_event_alpha.repeat(1,z_c.size()[1],1) # batch_size * dis_k * num_event
        # pos_event_alpha = self.alpha(pos_seqs[:, -2]) # batch_size * (dis_k * num_event)
        # pos_event_alpha = pos_event_alpha.view(z_c.size()[0],z_c.size()[1], -1) # batch_size * dis_k * num_event
        pos_event_alpha = alpha_adapt_gamma * pos_event_alpha + alpha_adapt_beta
        pos_event_delta = self.delta(pos_seqs[:, -2])  # batch_size * (dis_k * num_event)
        pos_event_delta = pos_event_delta.view(z_c.size()[0], z_c.size()[1],
                                               -1)  # batch_size * dis_k * num_event
        pos_event_delta = delta_adapt_gamma * pos_event_delta + delta_adapt_beta

        # import pdb
        # pdb.set_trace()
        # if torch.isnan(loss).any():

        # pos_alpha = (pos_event_alpha * torch.exp(-pos_event_delta * alpha_mask)).sum(dim=-1) # batch_size * dis_k
        pos_alpha = (pos_event_alpha, pos_event_delta, alpha_mask)

        # import pdb
        # pdb.set_trace()
        neg_event_alpha = self.event_emb.weight[neg_seqs[:, -2]].unsqueeze(1)  # batch_size * 1 * x_dim
        neg_event_all_alpha_T = self.event_emb.weight[:].transpose(0, 1).unsqueeze(0)  # 1 * x_dim * num_event
        neg_event_alpha = torch.matmul(neg_event_alpha, neg_event_all_alpha_T)  # batch_size * 1 * num_event
        neg_event_alpha = neg_event_alpha.repeat(1, z_c.size()[1], 1)  # batch_size * dis_k * num_event
        # neg_event_alpha = self.alpha(neg_seqs[:, -1])  # batch_size * (dis_k * num_event)
        # neg_event_alpha = neg_event_alpha.view(z_c.size()[0], z_c.size()[1], -1)  # batch_size * dis_k * num_event
        neg_event_alpha = alpha_adapt_gamma * neg_event_alpha + alpha_adapt_beta
        neg_event_delta = self.delta(neg_seqs[:, -1])  # batch_size * (dis_k * num_event)
        neg_event_delta = neg_event_delta.view(z_c.size()[0], z_c.size()[1], -1)  # batch_size * dis_k * num_event
        neg_event_delta = delta_adapt_gamma * neg_event_delta + delta_adapt_beta
        # neg_alpha = (neg_event_alpha * torch.exp(-neg_event_delta * alpha_mask)).sum(dim=-1)  # batch_size * dis_k
        neg_alpha = (neg_event_alpha, neg_event_delta, alpha_mask)
        return enc_output, pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, z_adapt, z_adapt_bar, kld_loss, p_time, mi_loss

    def test_seq2feats(self, event_input, time_input=[]):
        seqs_len = event_input.size()[1]
        index = torch.arange(0, seqs_len, 1)
        if self.opt["cuda"]:
            index = index.cuda()
        position_embedding = self.PE(index)
        position_embedding = position_embedding.unsqueeze(0)
        position_embedding = position_embedding.repeat(event_input.size()[0], 1, 1)
        biased_position_embedding = self.biasedPE(time_input)
        x = self.event_emb(event_input)
        phi_x = self.phi_x(x) + biased_position_embedding  # batch_size * seqs * x_dim

        z_c_x = torch.mean(phi_x, -2)  # batch_size * x_dim
        z_c_dis = self.z_c_x_to_dis(z_c_x)  # batch_size * dis_dim
        z_c_dis = z_c_dis.view(z_c_dis.size()[0], self.dis_k, -1)
        z_c_h = self.z_c_enc(z_c_dis)  # 128 * dis_k * h_dim
        z_c_mean = self.z_c_mean(z_c_h)  # batch_size * dis_k * z_dim
        z_c_std = self.z_c_std(z_c_h)  # batch_size * dis_k * z_dim
        z_c = self._reparameterized_sample(z_c_mean, z_c_std)  # batch_size * dis_k * z_dim
        z_c = self.phi_z_c(z_c)  # batch_size * dis_k * h_dim

        prior_h = Variable(torch.zeros(self.n_layers, x.size(0), self.x_dim).cuda())  # [layers, batch_size, hidden_dim]

        prior_h, last_prior_h = self.prior_rnn(phi_x, prior_h)
        last_prior_h = last_prior_h[-1]
        prior_h_dis = self.prior_fc(last_prior_h) # batch_size * x_dim
        prior_h_dis = prior_h_dis.view(prior_h_dis.size()[0], self.dis_k,
                                       -1)  # batch_size * dis_k * identity_dim
        prior_h_dis_to_zt = self.prior_enc(prior_h_dis)
        prior_mean = self.prior_mean(prior_h_dis_to_zt)
        prior_std = self.prior_std(prior_h_dis_to_zt)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean, prior_std) # batch_size * dis_k * z_dim
        phi_z_t = self.phi_z_t(z_t) # batch_size * dis_k * h_dim

        z = torch.cat((phi_z_t, z_c), dim = -1) # batch_size * dis_k * (2*h_dim)
        z_adapt = self.adapt(z).squeeze(-1)
        z_adapt_bar = self.adapt_bar(z).squeeze(-1)

        z_adapt = F.softmax(z_adapt, dim=-1)
        z_adapt_bar = F.softmax(z_adapt_bar, dim=-1)

        p_time_dis = self.time_predicter(z).squeeze(-1)  # batch_size * dis_k
        p_time_dis = p_time_dis * z_adapt_bar  # batch_size * dis_k
        p_time = torch.sum(p_time_dis, dim=-1)  # batch_size

        # decoder
        v_z_c = self.z_c_predicter(z_c.view(z_c.size()[0], -1))
        v_z_t = self.z_t_predicter(z_t.view(z_t.size()[0], -1))
        enc_output = self.event_predicter(torch.cat((last_prior_h, v_z_c, v_z_t),-1))  # batch_size * x_dim
        return enc_output, z_adapt, p_time

    def train_batch(self, event_seqs, time_seqs, pos_seqs, neg_seqs, alpha_mask, target_time): # for training
        enc_output, pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, z_adapt, z_adapt_bar, kld_loss, p_time, mi_loss = self.seq2feats(event_seqs, time_seqs, pos_seqs, neg_seqs, alpha_mask)
        # import pdb
        # pdb.set_trace()
        pos_embs = self.event_emb(pos_seqs[:,:-1])
        neg_embs = self.event_emb(neg_seqs[:,:-1])
        pos_logits = (enc_output * pos_embs).sum(dim=-1)
        neg_logits = (enc_output * neg_embs).sum(dim=-1)
        time_loss = ((p_time - target_time) * (p_time - target_time)).mean()
        return enc_output, pos_logits, neg_logits, kld_loss, pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, z_adapt, z_adapt_bar, time_loss, mi_loss # pos_pred, neg_pred

    def predict(self, event_seqs, time_seqs, rand_event_index): # for inference

        final_feat, z_adapt, p_time = self.test_seq2feats(event_seqs, time_seqs) # batch_size * dis_k * x_dim
        event_embs = self.event_emb(rand_event_index) # batch_size * seqs * x_dim
        # import pdb
        # pdb.set_trace()
        final_feat = final_feat.unsqueeze(1)
        final_feat = final_feat.repeat((1, event_embs.size()[1], 1))
        # logits = event_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits = (final_feat * event_embs).sum(dim=-1)

        return -logits, p_time  # preds # (U, I)


class PositionalEmbedding(nn.Module):

    def __init__(self, opt, max_len=4096):
        super().__init__()

        self.opt = opt

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, self.opt["x_dim"]).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.opt["x_dim"], 2).float() * -(math.log(10000.0) / self.opt["x_dim"])).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        if self.cuda:
            self.pe.cuda()

    def forward(self, x):
        return self.pe[x, :]


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, opt, max_len=4096):
        super().__init__()
        self.opt = opt
        self.sigm = nn.Sigmoid()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.opt["x_dim"], 2).float() * -(math.log(10000.0) / self.opt["x_dim"])).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)
        self.Wt = nn.Linear(1, self.opt["x_dim"] // 2, bias=False)
        if self.cuda:
            self.position.cuda()
            self.div_term.cuda()

    def forward(self, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        phi = self.sigm(phi)
        aa = len(interval.size())
        if aa > 1:
            length = interval.size(1)
        else:
            length = interval.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe