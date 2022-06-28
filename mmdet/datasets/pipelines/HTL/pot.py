import numpy as np
import cv2
import os
def compute_prob(h,w,gt_boxes):
    #import pdb;pdb.set_trace()
    prob_map = np.zeros([h,w])
    for gt_box in gt_boxes:
        xmin = int(gt_box[0])
        xmax = int(gt_box[2])
        ymin = int(gt_box[1])
        ymax = int(gt_box[3])
        if (xmax - xmin)<1:
            xmax +=1
            print('width of box is set to be 1')
        if (ymax - ymin)<1:
            ymax +=1
            print('height of box is set to be 1')        
        for x in range(xmin,xmax):
            for y in range(ymin,ymax):
                l_r_min = min(x - xmin,xmax-x)
                l_r_max = max(x - xmin,xmax-x)
                b_t_min = min(y - ymin,ymax-y)
                b_t_max = max(y - ymin,ymax-y)
                prob_map[y,x] = np.clip(prob_map[y,x]  + np.sqrt((l_r_min*b_t_min+ 1e-10)/(l_r_max*b_t_max+ 1e-10)),0,1)
    return prob_map
def save_density_map(density_map, output_dir = '/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/results.png'):

    density_map = 255.0 * (density_map - np.min(density_map) + 1e-10) / (1e-10 + np.max(density_map) - np.min(density_map))
    density_map = density_map.squeeze()
    color_map = cv2.applyColorMap(density_map[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir), color_map)

if __name__ == "__main__":
    #import pdb;pdb.set_trace()
    
    gt_boxes = np.array([[10,50,80,100],[100,300,200,400]])
    prob_map = compute_prob(prob_map,gt_boxes)
    save_density_map(prob_map)

