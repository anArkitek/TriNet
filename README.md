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

## Video demo
[video](https://www.youtube.com/watch?v=kYeEM_WB_DI)

## Dataset
Training size: 10075 <br>
Validation size: 1120 <br>
<br>
raw video: 1280 * 720    40FPS
processed image(remove distortion): 960 * 720

network input: 244 * 244

## One model for all(degree error)
|#|Experiment|front vector loss(training)|right vector loss(training)|up vector loss(training)|validation loss(total)|
| :--- | :----: | ----: |----: |----: |----: |
|1|one model for three vectors|3.767|3.375|3.732|11.195|
|2|one model for three vectors with constraints|3.109|3.184|3.187|11.233|

<br>

## One model for all(training loss)
|#|Experiment|front vector loss(training)|right vector loss(training)|up vector loss(training)|
| :--- | :----: | ----: |----: |----: |
|1|one model for three vectors|0.232|0.230|0.231|
|2|one model for three vectors with constraints|0.232|0.233|0.229|

## Pareto Plot on error
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/right_error.png" width="800">


