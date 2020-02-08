import cv2
import numpy as np
import os

preds_imgs = './visualization/show_front'

img_list = os.listdir(preds_imgs)
imgs = [os.path.join(preds_imgs, img) for img in img_list]
imgs_obj = [cv2.imread(x) for x in imgs]

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def all_concat(objs,interpolation=cv2.INTER_CUBIC):
    r = int(len(objs)**0.5)
    objs_ = []
    for i in range(0,len(objs),r):
          objs_.append(objs[i:i+4])
    return cv2.hconcat(objs_)

im_v_resize = all_concat(imgs_obj)
cv2.imwrite('test.jpg', im_v_resize)

print("Done combining.")


