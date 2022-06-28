from cv2 import imread, imwrite
import torch
from torch.autograd import Variable
from .net_canny import Net
import numpy as np
import os
import pdb
def edge_gen(raw_img,annots,use_cuda=False):
    early_threshold, _ = canny_gen(raw_img/255.0)
    _, edge = canny_gen(np.repeat(early_threshold[...,np.newaxis],3,2)/255)

    mask = np.zeros((edge.shape[0],edge.shape[1]))

    for box in annots:
        mask[int(box[1]):int((box[3]+1))][:,int(box[0]):int((box[2]+1))] = 1
    edge_mask = edge * mask

    return edge,edge_mask,mask


def canny_gen(raw_img,use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    net = Net(threshold=1.5, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    
    thresholded,early_threshold = net(data)
    edge = (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)
    early = early_threshold.data.cpu().numpy()[0, 0]


    return early,edge


if __name__ == '__main__':
    image_path = '/share/home/zhangxin/data/LS-SSDD-v1.0-OPEN/JPEGImages/'
    imgs = os.listdir(image_path)
    #import pdb;pdb.set_trace()
    for img in imgs:
        image_name = image_path+img
        im = imread(image_name) / 255.0

    # canny(img, use_cuda=False)
        edge,edge_mask,mask = edge_gen(im, image_name,use_cuda=True)

        edge_save_name = '/share/home/zhangxin/data/LS-SSDD-v1.0-OPEN/edge_ori/' + img.split('.')[0] + '.png' # true edge

        cv2.imwrite(edge_save_name,edge_mask*255)
