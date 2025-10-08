import torch.nn as nn
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")









## KD

def KL_Div(s, t, T=6.0):
    x = F.log_softmax(s/T, dim=1)
    y = F.softmax(t/T, dim=1)
    KLDiv = F.kl_div(x,y, reduction='batchmean')
    return KLDiv

def KD_Loss(y, s, t, criterion, T=1.0, alpha=0.7):
    l_ce  = criterion(s, y)
    l_kl = KL_Div(s, t, T)

    KDLoss = (1.0 - alpha) * l_ce  + (alpha * T * T) * l_kl
    return KDLoss


## AT

def single_stage_at_loss(f_s, f_t):
    def _at(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def at_loss(f_s, f_t):
        return (_at(f_s) - _at(f_t)).pow(2).mean()

    return at_loss(f_s,f_t)

def at_loss(g_s, g_t):
    return sum([single_stage_at_loss(f_s, f_t) for f_s, f_t in zip(g_s.values(), g_t.values())])


##SP similarity preserving

def single_stage_sp_loss(f_s, f_t):


    def _sp(x):
        Q = x.view(x.size(0), -1)
        G = F.normalize(torch.mm(Q, Q.permute(1, 0)))
        return G


    def sp_loss(x, y):
        return (_sp(x) - _sp(y)).pow(2).mean()

    return sp_loss(f_s,f_t)

def sp_loss(g_s, g_t):
    return sum([single_stage_sp_loss(f_s, f_t) for f_s, f_t in zip(g_s.values(), g_t.values())])





class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Convert alpha to tensor if it's a list
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):

        log_probs = F.log_softmax(inputs, dim=1)

        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # Probability of true class
        log_pt = (log_probs * targets_one_hot).sum(dim=1)  # Log prob of true class

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # More efficient than gather
            focal_weight = at * ((1 - pt) ** self.gamma)
        else:
            focal_weight = (1 - pt) ** self.gamma

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DistillationSPLoss(nn.Module):
    def __init__(self, class_weights,alpha):
        super(DistillationSPLoss, self).__init__()
        self.class_weights = class_weights
        self.standard_loss = FocalLoss(gamma=2., alpha=class_weights)
        self.alpha = alpha


    def forward(self,logits_s,feats_s,feats_t,target):




        loss_feat = self.alpha* sp_loss(feats_s,feats_t)

        hard_loss = (1-self.alpha) * self.standard_loss(logits_s, target)


        loss = hard_loss+loss_feat

        return loss

class DistillationATLoss(nn.Module):
    def __init__(self, class_weights,alpha):
        super(DistillationATLoss, self).__init__()
        self.class_weights = class_weights
        self.standard_loss = FocalLoss(gamma=2., alpha=class_weights)
        self.alpha = alpha


    def forward(self,logits_s,feats_s,feats_t,target):




        loss_feat = self.alpha* at_loss(feats_s,feats_t)

        hard_loss = (1-self.alpha) * self.standard_loss(logits_s, target)


        loss = hard_loss+loss_feat

        return loss

class DistillationKLDivLoss(nn.Module):
    def __init__(self, temperature, alpha, class_weights):
        super(DistillationKLDivLoss, self).__init__()
        self.class_weights = class_weights
        self.standard_loss = FocalLoss(gamma=2., alpha=class_weights)

        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
        self.alpha = alpha

    def forward(self,student_logits, teacher_logits, target):
        # Standard Focal loss
        hard_loss = self.standard_loss(student_logits, target)

        # Distillation loss
        soft_loss = self.soft_loss(self.log_softmax(student_logits / self.temperature),
                                                  self.softmax(teacher_logits / self.temperature))

        soft_loss = soft_loss * self.temperature ** 2
        KDLoss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return KDLoss






class_weights = torch.load("class_weights.pt",weights_only=True).tolist()
criterion = FocalLoss(gamma=2., alpha=class_weights)
KD = DistillationKLDivLoss(7, 0.6, class_weights)

AT = DistillationATLoss(class_weights,0.6)

SP = DistillationSPLoss(class_weights,0.6)