# TriNet for object pose estimation
3 Dof object pose estimation with new representation

## Train

python train.py  --num_classes [33,66] --num_epochs --lr --lr_decay --unfreeze 

--train_data --valid_data --input_size [224,196,160,128,96] 

--width_mult [1.0,0.5] --batch_size --top_k --cls2reg --alpha

--save_dir

batch size: 64
epoches: 50
learning rate: 0.01
lr_decay: 0.8

## Test on testing dataset

python test.py --snapshot --analysis

## Single image testing

python test_on_img.py --img --snapshot

## Video testing

python video_demo.py --video --snapshot


## Dataset
Training size: 10075 <br>
Validation size: 1120 <br>
<br>
raw video: 1280 * 720    40FPS
processed image(remove distortion): 960 * 720

network input: 244 * 244



