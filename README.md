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
[video](https://www.youtube.com/watch?v=BvXT__mZwwM)

## Dataset
Training size: 14728 <br>
Validation size: 1646 <br>
<br>
raw video: 1280 * 720    40FPS
processed image(remove distortion): 960 * 720

network input: 244 * 244

## One model for all(degree error)
|#|Experiment|front vector error(training)|right vector error(training)|up vector error(training)|validation error(total)|
| :--- | :----: | ----: |----: |----: |----: |
|1|one model for three vectors|3.767|3.375|3.732|11.195|
|2|one model for three vectors with constraints(a = 0.1)|2.450|2.395|2.415|8.327|
|3|one model for three vectors with constraints(a = 0.075)|2.242|2.401|2.462|7.319|

<br>

## One model for all(training loss)
|#|Experiment|front vector loss(training)|right vector loss(training)|up vector loss(training)|
| :--- | :----: | ----: |----: |----: |
|1|one model for three vectors|0.232|0.230|0.231|
|2|one model for three vectors with constraints(a = 0.1)|0.230|0.228|0.227|
|3|one model for three vectors with constraints(a = 0.075)|0.229|0.226|0.226|

## Pareto Plot on angle errors
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/front_error.png" width="600">
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/right_error.png" width="600">
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/up_error.png" width="600">

### Scatter plot on Euler-angle error
<img src="" width = 600>

## Test
### Video test demo(no tracker)
[video](https://www.youtube.com/watch?v=1aavWYp1kSg)

