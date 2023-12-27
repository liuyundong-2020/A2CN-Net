# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np
from utils.metrics import bbox_iou, Wasserstein,box_iou1
from utils.torch_utils import de_parallel
import math
Slide_Loss=True
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
# reference: https://github.com/dongdonghy/repulsion_loss_pytorch/blob/master/repulsion_loss.py
def IoG(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def smooth_ln(x, deta=0.5):
    return torch.where(
        torch.le(x, deta),
        -torch.log(1 - x),
        ((x - deta) / (1 - deta)) - np.log(1 - deta)
    )

# YU 添加了detach，减小了梯度对gpu的占用
def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    pgiou = box_iou1(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = pgiou.cuda().data.cpu().numpy()
    ppiou = box_iou1(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = ppiou.cuda().data.cpu().numpy()
    # t1 = time.time()
    len = pgiou.shape[0]
    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
            # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
            # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
            if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0

    # t2 = time.time()
    # print("for cycle cost time is: ", t2 - t1, "s")
    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()
    # repgt
    max_iou, argmax_iou = torch.max(pgiou, 1)
    pg_mask = torch.gt(max_iou, gtnms)
    num_repgt = pg_mask.sum()
    if num_repgt > 0:
        iou_pos = pgiou[pg_mask, :]
        max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
        pbox_sec = pbox[pg_mask, :]
        gtbox_sec = gtbox[argmax_iou_sec, :]
        IOG = IoG(gtbox_sec, pbox_sec)
        repgt_loss = smooth_ln(IOG, deta)
        repgt_loss = repgt_loss.mean()

    # repbox
    pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
    # print(mem)
    torch.cuda.empty_cache()

    return repgt_loss, repbox_loss
class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = True


    # Compute losses
    def __init__(self, model, autobalance=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.BCEclsCost = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction="none")
        self.BCEobjCost = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction="none")

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        if Slide_Loss:
            BCEcls, BCEobj = SlideLoss(BCEcls), SlideLoss(BCEobj)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lrepBox, lrepGT = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            tbx = tbox[i]
            anc = anchors[i]
            tc = tcls[i]

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0

                temps_s = b * 3 * 80 * 81 + a * 80 * 81 + gj * 81 + gi + 1  # 1228800 = 64 * 3 * 80 * 80
                temps_lst, temps_idx = temps_s.sort(dim=-1)

                temps_lst2 = torch.zeros_like(temps_lst, device=self.device)
                temps_lst2[1:] = temps_lst[:n - 1]
                temps_lst = (temps_lst - temps_lst2) > 0
                head_lst = torch.ones_like(temps_lst, device=self.device)
                head_lst[:n - 1] = temps_lst[1:]
                head_lst = ((temps_lst.int() - head_lst.int()) > 0).int()

                r_lst = torch.zeros_like(temps_lst, device=self.device)
                r_lst[:n - 1] = (~temps_lst)[1:]
                r_lst = (~((temps_lst.int() - r_lst.int()) > 0)).int()

                al_lst = head_lst + r_lst

                al_lst_single = al_lst == 0
                al_lst_single = torch.nonzero(al_lst_single, as_tuple=True)

                al_lst_idx = torch.nonzero(al_lst, as_tuple=True)

                new_idx = temps_idx[al_lst_idx]

                bps = pi[b[new_idx], a[new_idx], gj[new_idx], gi[new_idx]]
                bpxy = bps[:, :2].sigmoid() * 2 - 0.5
                bpwh = (bps[:, 2:4].sigmoid() * 2) ** 2 * anc[new_idx]
                bpbox = torch.cat((bpxy, bpwh), 1)  # predicted box
                biou = bbox_iou(bpbox.T, tbx[new_idx], xywh=False, CIoU=True)  # iou(prediction, target)
                #iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                cost_reg = (1.0 - biou)  # iou loss
                bt = torch.full_like(bps[:, 5:], self.cn, device=self.device)  # targets
                bt[range(len(new_idx)), tc[new_idx]] = self.cp
                cost_cls = self.BCEclsCost(bps[:, 5:], bt).sum(dim=-1)  # BCE

                # bo = torch.ones_like(bps[:, 4], device=device)
                # cost_obj = self.BCEobjCost(bps[:, 4], bo)  # BCE

                cost = cost_cls + cost_reg * 3.  # + cost_obj * 3.

                al_s = 0
                final_idx = []
                al_total = al_lst[al_lst_idx].shape[0]
                for ali, al in enumerate(al_lst[al_lst_idx]):
                    if al == 2:
                        if ali != 0:  # and ali - al_s > 1:
                            # print(1,biou[al_s:ali].shape, cost[al_s:ali].shape, ali - al_s)
                            # if ali - al_s>1:
                            _, bl_idx = biou[al_s:ali].topk(k=2)
                            _, al_idx = cost[al_s:ali][bl_idx].min(dim=-1)
                            final_idx.append(new_idx[al_s + bl_idx[al_idx]].item())
                            # else:
                            #     final_idx.append(new_idx[al_s : ali].item())
                            al_s = ali
                    elif ali == al_total - 1:  # and al_total - al_s > 1:
                        # print(2,biou[al_s:].shape,al_total - al_s)
                        # if al_total - al_s>1:
                        _, bl_idx = biou[al_s:].topk(k=2)
                        _, al_idx = cost[al_s:][bl_idx].min(dim=-1)
                        final_idx.append(new_idx[al_s + bl_idx[al_idx]].item())
                        # else:
                        #     final_idx.append(new_idx[al_s: ].item())
                    # elif ali - al_s == 1 or al_total - al_s == 1:
                    #     print(3,biou[al_s:ali].shape)
                    #     final_idx.append(new_idx[ali].item())
                    #     al_s = ali

                if len(final_idx) > 0:
                    try:
                        final_idx = torch.tensor(final_idx, device=device)
                        final_idx = torch.cat([final_idx, temps_idx[al_lst_single]], dim=-1)
                        b, a, gj, gi = b[final_idx], a[final_idx], gj[final_idx], gi[final_idx]
                        tbx, anc, tc, n = tbx[final_idx], anc[final_idx], tc[final_idx], final_idx.shape[0]
                    except:
                        print(final_idx)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                auto_iou = iou.mean()
                nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, tbox[i], x1y1x2y2=False), 1 / 2) / 1.0)
                lbox += 0.8 * (1.0 - iou).mean() + 0.2 * (1.0 - nwd).mean()

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls[:, :], t, auto_iou)  # BCE

                dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
                       13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
                       25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [],
                       37: [], 38: [], 39: [], 40: [], 41: [], 42: [], 43: [], 44: [], 45: [], 46: [], 47: [], 48: [],
                       49: [], 50: [], 51: [], 52: [], 53: [], 54: [], 55: [], 56: [], 57: [], 58: [], 59: [], 60: [],
                       61: [], 62: [], 63: [], 64: [], 65: [], 66: [], 67: [], 68: [], 69: [], 70: [], 71: [], 72: [],
                       73: [], 74: [], 75: [], 76: [], 77: [], 78: [], 79: [], 80: [], 81: [], 82: [], 83: [], 84: [],
                       85: [], 86: [], 87: [], 88: [], 89: [], 90: [], 91: [], 92: [], 93: [], 94: [], 95: [], 96: [],
                       97: [], 98: [], 99: [], 100: [], 101: [], 102: [], 103: [], 104: [], 105: [], 106: [], 107: [],
                       108: [], 109: [], 110: [], 111: [], 112: [], 113: [], 114: [], 115: [], 116: [], 117: [],
                       118: [],
                       119: [], 120: [], 121: [], 122: [], 123: [], 124: [], 125: [], 126: [], 127: [], 128: [],
                       129: [],
                       130: [], 131: [], 132: [], 133: [], 134: [], 135: [], 136: [], 137: [], 138: [], 139: [],
                       140: [],
                       141: [], 142: [], 143: [], 144: [], 145: [], 146: [], 147: [], 148: [], 149: []}
                for indexs, value in enumerate(b):
                    # print(indexs, value)
                    dic[int(value)].append(indexs)
                # print('dic', dic)
                bts = 0
                deta = 0.5
                Rp_nms = 0.1
                _lrepGT = 0.0
                _lrepBox = 0.0
                for id, indexs in dic.items():  # id = batch_name  indexs = target_id
                    if indexs:
                        lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], tbox[i][indexs], deta=deta, pnms=Rp_nms,
                                                               gtnms=Rp_nms)
                        _lrepGT += lrepgt
                        _lrepBox += lrepbox
                        bts += 1
                if bts > 0:
                    _lrepGT /= bts
                    _lrepBox /= bts
                lrepGT += _lrepGT
                lrepBox += _lrepBox

            if n:
                obji = self.BCEobj(pi[..., 4], tobj, auto_iou)
            else:
                obji = self.BCEobj(pi[..., 4], tobj)

            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lrep = 0.01 * lrepGT / 3.0 + 0.1 * lrepBox / 3.0
        bs = tobj.shape[0]  # batch size
        loss = lbox + lobj + lcls + lrep
        return loss * bs, torch.cat((lbox, lobj, lcls, lrep, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
