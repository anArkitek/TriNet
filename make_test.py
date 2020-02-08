import os
import shutil
import random

test_img = './dataset/bg_imgs'

test_info = './info_all'

dst_img = './test_imgs'
dst_info = './test_info'

name_list = sorted(os.listdir(test_img))
randoms = random.choices(name_list,k=32)

for name in randoms:
	shutil.copy(os.path.join(test_img,name),dst_img)
	shutil.copy(os.path.join(test_info,name.split('.')[0]+'.txt'),dst_info)
	print(name+ " "+os.path.join(test_info,name.split('.')[0]+'.txt'))

print("Done copying.")

