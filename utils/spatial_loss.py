import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Spatial_Loss(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Loss, self).__init__()
        self.res_scale = in_channels
        
        self.make_PAN = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        self.L1_loss = nn.L1Loss()
        
    def forward(self, ref_HS, pred_HS):
        pan_pred = self.make_PAN(pred_HS)
        with torch.no_grad():
            pan_ref = self.make_PAN(ref_HS)     
        spatial_loss = self.L1_loss(pan_pred, pan_ref.detach())
        return spatial_loss
    
    
class Edge_spec_Loss(nn.Module):
    "https://doi.org/10.1111/phor.70019"
    def __init__(self):
        super(Edge_spec_Loss, self).__init__()
        
    def forward(self, ref_HS, pred_HS, edge_mask):
        b, c, h, w = ref_HS.shape
        edge_weight = F.sigmoid(edge_mask.view(b, 1, -1))
        ref_HS = ref_HS.reshape(b, c, -1)
        pred_HS = pred_HS.reshape(b, c, -1)
        batch_loss = edge_weight* torch.sum(torch.abs(ref_HS-pred_HS), dim=1)/c
        #print(batch_loss.mean())
        return batch_loss.mean()
    
    
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return 
    
    
class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
        
    def forward(self, HS_feats, MS_feats):
        b, c, H, W = HS_feats.shape
       
        HS_feats = HS_feats.reshape(b, c, -1)
        MS_feats = MS_feats.reshape(b, c, -1)
        batch_loss = torch.sqrt(torch.sum((HS_feats-MS_feats)**2, dim=1))/c
        #print(batch_loss.mean())
        return batch_loss.mean()
    
class DKL(nn.Module):
    def __init__(self):
        super(DKL, self).__init__()
        
    def forward(self, student_probs, teacher_probs):
        logp_x = F.log_softmax(student_probs, dim=1)
        p_y = F.softmax(teacher_probs, dim=1)
        kl_divergence = F.kl_div(logp_x, p_y, reduction='batchmean')
        
        return kl_divergence.mean()