import torch
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate(feature_stractor, classifier, data_loader):
    ''' set model to evaluate mode '''
    feature_stractor.eval()
    classifier.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
                imgs = imgs.cuda()
                fea = feature_stractor(imgs)
                _, pred = classifier(fea)

                _, pred = torch.max(pred, dim=1)

                pred = pred.cpu().numpy().squeeze()
                gt = gt.numpy().squeeze()

                preds.append(pred)
                gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)
