import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class GeneralizedIOU(nn.Module):
    """
    Implemented from g-darknet
    https://github.com/generalized-iou/g-darknet/blob/1421328ef53859a94fe8c434565e4838c3a7b03d/scripts/iou_utils.py
    """

    def __init__(self):
        pass

    def forward(self):
        pass

    def intersection(self, a, b):
        '''
            input: 2 boxes (a,b)
            output: overlapping area, if any
        '''
        top = torch.max(a[0], b[0])
        left = torch.max(a[1], b[1])
        bottom = torch.min(a[2], b[2])
        right = torch.min(a[3], b[3])
        h = torch.max(bottom - top, 0)
        w = torch.max(right - left, 0)
        return h * w

    def union(self, a, b):
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        return a_area + b_area - self.intersection(a, b)

    def c(self, a, b):
        '''
            input: 2 boxes (a,b)
            output: smallest enclosing bounding box
        '''
        top = torch.min(a[0], b[0])
        left = torch.min(a[1], b[1])
        bottom = torch.max(a[2], b[2])
        right = torch.max(a[3], b[3])
        h = torch.max(bottom - top, 0)
        w = torch.max(right - left, 0)
        return h * w

    def iou(self, a, b):
        '''
            input: 2 boxes (a,b)
            output: Itersection/Union
        '''
        U = self.union(a, b)
        if U == 0:
            return 0
        return intersection(a, b) / U

    def giou(self, a, b):
        '''
            input: 2 boxes (a,b)
            output: Itersection/Union - (c - U)/c
        '''
        I = self.intersection(a, b)
        U = self.union(a, b)
        C = self.c(a, b)
        iou_term = (I / U) if U > 0 else 0
        giou_term = ((C - U) / C) if C > 0 else 0
        # print("  I: %f, U: %f, C: %f, iou_term: %f, giou_term: %f"%(I,U,C,iou_term,giou_term))
        return iou_term - giou_term


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=0.25):
#         """
#             focusing is parameter that can adjust the rate at which easy
#             examples are down-weighted.
#             alpha may be set by inverse class frequency or treated as a hyper-param
#             If you don't want to balance factor, set alpha to 1
#             If you don't want to focusing factor, set gamma to 1
#             which is same as normal cross entropy loss
#         """
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#
#     def forward(self, predictions, targets):
#         """
#             Args:
#                 predictions (tuple): (conf_preds, loc_preds)
#                     conf_preds shape: [batch, n_anchors, num_cls]
#                     loc_preds shape: [batch, n_anchors, 4]
#                 targets (tensor): (conf_targets, loc_targets)
#                     conf_targets shape: [batch, n_anchors]
#                     loc_targets shape: [batch, n_anchors, 4]
#         """
#
#         conf_preds, loc_preds = predictions
#         conf_targets, loc_targets = targets
#
#         ############### Confiden Loss part ###############
#         """
#         #focal loss implementation(1)
#         pos_cls = conf_targets > -1 # exclude ignored anchors
#         mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
#         conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
#         conf_t = conf_targets[pos_cls].view(-1).clone()
#         p = F.softmax(conf_p, 1)
#         p = p.clamp(1e-7, 1. - 1e-7) # to avoid loss going to inf
#         c_mask = conf_p.data.new(conf_p.size(0), conf_p.size(1)).fill_(0)
#         c_mask = Variable(c_mask)
#         ids = conf_t.view(-1, 1)
#         c_mask.scatter_(1, ids, 1.)
#         p_t = (p*c_mask).sum(1).view(-1, 1)
#         p_t_log = p_t.log()
#         # This is focal loss presented in ther paper eq(5)
#         conf_loss = -self.alpha * ((1 - p_t)**self.gamma * p_t_log)
#         conf_loss = conf_loss.sum()
#         """
#
#         conf_targets = self.one_hot(conf_targets, 80)
#         # focal loss implementation(2)
#         pos_cls = conf_targets > -1
#         mask = pos_cls.unsqueeze(2).expand_as(conf_preds)
#         conf_p = conf_preds[mask].view(-1, conf_preds.size(2)).clone()
#         p_t_log = -F.cross_entropy(conf_p, conf_targets[pos_cls], size_average=False)
#         p_t = torch.exp(p_t_log)
#
#         # This is focal loss presented in the paper eq(5)
#         conf_loss = -self.alpha * ((1 - p_t) ** self.gamma * p_t_log)
#
#         ############# Localization Loss part ##############
#         pos = conf_targets > 0  # ignore background
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
#         loc_p = loc_preds[pos_idx].view(-1, 4)
#         loc_t = loc_targets[pos_idx].view(-1, 4)
#         loc_loss = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
#
#         num_pos = pos.long().sum(1, keepdim=True)
#         N = max(num_pos.data.sum(),
#                 1)  # to avoid divide by 0. It is caused by data augmentation when crop the images. The cropping can distort the boxes
#         conf_loss /= N  # exclude number of background?
#         loc_loss /= N
#
#         return conf_loss, loc_loss
#
#     def one_hot(self, x, n):
#         y = torch.eye(n)
#         return y[x]


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(DEVICE))
                classification_losses.append(torch.tensor(0).float().to(DEVICE))

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.to(DEVICE)

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).to(DEVICE) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(DEVICE))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(DEVICE)

                negative_indices = 1 - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(DEVICE))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0,
                                                                                                                 keepdim=True)
