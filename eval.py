import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score




def feat_get(step, G, Cs, dataset_source, dataset_target, save_path,
             ova=True):
    G.eval()

    for batch_idx, data in enumerate(dataset_source):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_s = data[0]
            label_s = data[1]
            img_s, label_s = Variable(img_s.cuda()), \
                             Variable(label_s.cuda())
            feat_s = G(img_s)


            if batch_idx == 0:
                feat_all_s = feat_s.data.cpu().numpy()
                label_all_s = label_s.data.cpu().numpy()
            else:
                feat_s = feat_s.data.cpu().numpy()
                label_s = label_s.data.cpu().numpy()
                feat_all_s = np.r_[feat_all_s, feat_s]
                label_all_s = np.r_[label_all_s, label_s]
    for batch_idx, data in enumerate(dataset_target):
        if batch_idx == 500:
            break
        with torch.no_grad():
            img_t = data[0]
            label_t = data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat_t = G(img_t)

            out_t = Cs[0](feat_t)
            pred = out_t.data.max(1)[1]
            out_t = F.softmax(out_t)
            if ova:
                out_open = Cs[1](feat_t)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1), 1)
                tmp_range = torch.range(0, out_t.size(0) - 1).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                weights_open = Cs[1].module.fc.weight.data.cpu().numpy()
            else:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)

            if batch_idx == 0:
                feat_all = feat_t.data.cpu().numpy()
                label_all = label_t.data.cpu().numpy()
                unk_all = pred_unk.data.cpu().numpy()
                pred_all = pred.data.cpu().numpy()
                pred_all_soft = out_t.data.cpu().numpy()
            else:
                feat_t = feat_t.data.cpu().numpy()
                label_t = label_t.data.cpu().numpy()
                pred_unk = pred_unk.data.cpu().numpy()
                feat_all = np.r_[feat_all, feat_t]
                label_all = np.r_[label_all, label_t]
                unk_all = np.r_[unk_all, pred_unk]
                pred_all = np.r_[pred_all, pred.data.cpu().numpy()]
                pred_all_soft = np.r_[pred_all_soft, out_t.data.cpu().numpy()]

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "save_%s_ova_%s_target_feat.npy" % (step, ova)), feat_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_anom.npy" % (step, ova)), unk_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_pred.npy" % (step, ova)), pred_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_soft.npy" % (step, ova)), pred_all_soft)
    np.save(os.path.join(save_path, "save_%s_ova_%s_source_feat.npy" % (step, ova)), feat_all_s)
    np.save(os.path.join(save_path, "save_%s_ova_%s_target_label.npy" % (step, ova)), label_all)
    np.save(os.path.join(save_path, "save_%s_ova_%s_source_label.npy" % (step, ova)), label_all_s)
    if ova:
        np.save(os.path.join(save_path, "save_%s_ova_%s_weight.npy" % (step, ova)), weights_open)




def test(step, dataset_test, name, n_share, G, Cs,
         open_class = None, open=False, entropy=False, thr=None):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.range(0, out_t.size(0)-1).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]
            pred[ind_unk] = open_class
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        roc = roc_auc_score(Y_test[:, -1], pred_open)
        roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_acc, mean_score = select_threshold(pred_all, pred_open,
                                                         label_all, class_list)

    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s"%(float(per_class_acc.mean())),
              "acc %s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "h score %s" % float(h_score),
              "roc %s"% float(roc),
              "roc ent %s"% float(roc_ent),
              "roc softmax %s"% float(roc_softmax),
              "best hscore %s"%float(best_acc),
              "best thr %s"%float(best_th)]
    logger.info(output)
    print(output)
    return acc_all, h_score


def select_threshold(pred_all, conf_thr, label_all,
                     class_list, thr=None):
    num_class  = class_list[-1]
    best_th = 0.0
    best_f = 0
    #best_known = 0
    if thr is not None:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > thr)[0]
        pred_class[ind_unk] = num_class
        return accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class)
    ran = np.linspace(0.0, 1.0, num=20)
    conf_thr = conf_thr / conf_thr.max()
    scores = []
    for th in ran:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > th)[0]
        pred_class[ind_unk] = num_class
        score, known, unknown = h_score_compute(label_all, pred_class,
                                                class_list)
        scores.append(score)
        if score > best_f:
            best_th = th
            best_f = score
            best_known = known
            best_unknown = unknown
    mean_score = np.array(scores).mean()
    print("best known %s best unknown %s "
          "best h-score %s"%(best_known, best_unknown, best_f))
    return best_th, best_f, mean_score


def h_score_compute(label_all, pred_class, class_list):
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros((len(class_list))).astype(np.float32)
    for i, t in enumerate(class_list):
        t_ind = np.where(label_all == t)
        correct_ind = np.where(pred_class[t_ind[0]] == t)
        per_class_correct[i] += float(len(correct_ind[0]))
        per_class_num[i] += float(len(t_ind[0]))
    open_class = len(class_list)
    per_class_acc = per_class_correct / per_class_num
    known_acc = per_class_acc[:open_class - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    return h_score, known_acc, unknown
