import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.D2MHP.D2MHP import D2MHP
from model.D2MHP import Utils
class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class PPTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if opt["model"] == "d2mhp":
            self.model = D2MHP(opt)
        else:
            print("please select a valid model")
            exit(0)

        self.bce_criterion = nn.BCEWithLogitsLoss()
        if opt['cuda']:
            for name, param in self.model.named_parameters():
                try:
                    torch.nn.init.xavier_uniform_(param.data)
                except:
                    pass
            self.model.cuda()
            self.bce_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.epoch_pred_loss = 0
        self.epoch_kl_loss = 0
        self.epoch_event_loss = 0
        self.epoch_time_loss = 0
        self.epoch_mi_loss = 0

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        return user_index, item_index, negative_item_index

    def train_batch(self, seq, time_seq, pos, neg, alpha_mask, target_time=None):
        self.model.train()
        self.optimizer.zero_grad()

        enc_output, pos_logits, neg_logits, kld_loss, pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, z_adapt, z_adapt_bar, time_loss, mi_loss = self.model.train_batch(seq, time_seq, pos, neg, alpha_mask, target_time)

        pos_labels, neg_labels = torch.ones(pos_logits.size()).cuda(), torch.zeros(neg_logits.size()).cuda()
        pred_loss = self.bce_criterion(pos_logits, pos_labels)
        pred_loss += self.bce_criterion(neg_logits, neg_labels)

        event_loss = Utils.log_likelihood_dvhp_adapt(pos_dis_mu, neg_dis_mu, pos_alpha, neg_alpha, z_adapt, z_adapt_bar, self.opt)

        loss = pred_loss + kld_loss + event_loss + self.opt["mi_loss"]*mi_loss
        # loss = pred_loss + kld_loss + event_loss
        self.epoch_pred_loss += pred_loss.item()
        self.epoch_kl_loss += kld_loss.item()
        self.epoch_event_loss += event_loss.item()
        self.epoch_time_loss += time_loss.item()
        self.epoch_mi_loss += mi_loss.item()
        loss.backward()

        """ update parameters """
        self.optimizer.step()
        return loss.item()