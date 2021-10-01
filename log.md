### 27-28 May, 2021
Found a new architecture from the [benchmark](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-cityscapes):  

Paper: [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/abs/2101.06085v1)  
Github repo: [DDRNet-23-slim](https://github.com/ydhongHIT/DDRNet) pre-trained model included  
Architecture Github repo 1: [DDRNet](https://github.com/chenjun2hao/DDRNet.pytorch)  
Architecture Github repo 2: [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)  
Rank 1st of mIoU  
Rank 3rd of Time  

Comparison:  
![benchmark_index](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/benchmark_index.png)
![benchmark_DDRNet](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/benchmark_DDRNet.png)
![benchmark_FRRN](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/benchmark_FRRN.png)

#### Test on our images
Use the first architecture [DDRNet](https://github.com/chenjun2hao/DDRNet.pytorch)  
Config file: ```${ProjectPath}/experiments/cityscapes/ddrnet23.yaml```  
Pretrained model: [Download](https://drive.google.com/file/d/16viDZhbmuc3y7OSsUo2vhA7V6kYO0KX6/view)  
Create a test scripts: Upload later..  

Test image:  
![test image](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/camera1599827799004.jpeg)  
~~Since our image size is 320\*1900 (h*w), CityScapes size is 1024\*2048, it should preprocess the image. Options:~~  
(Fully convolutional network)  
~~1. Resize the image~~
![resize](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/resize.jpeg)
![resize result](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/resize_result.png)
~~2. Pad the image~~
![pad](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/pad.jpeg)
![pad result](https://github.com/MingZx8/UrbanScannerAnnotation/blob/main/example/pad_result.png)


### 11-14 June, 2021
##### 300 labelled images  
Check and caclulate ratio of labelled classes in the images
+ unclassified polygon

### July-August, 2021
#### Real-time object detection on COCO: [YOLO4](https://github.com/AlexeyAB/darknet) 
1. clone repo  
2. modify 'Makefile'  
  a. `GPU=1` to enable GPU  
  b. `CUDNN=1` to enable CUDA (CUDA = 10.1 on Ubuntu 18.04, `export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64 && sudo ldconfig`)  
  c. `CUDNN_HALF=1` for acceleration  
  d. `OPENCV=1` to enable opencv  
  e. `NVCC=/usr/local/cuda/bin/nvcc`  
3. go to `src/image.c`  
  a. add `printf("%d %d %d %d\n",x1, x2, y1, y2);` to `void draw_box_width` to print bbox  
4. ```./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show < test.txt > result.txt```
5. ```./darknet detector train data/obj.data cfg/yolov4x-mish_custom.cfg yolov4-obj_best.weights -map -gpus 02>&1 | tee -a log_mish.txt```

#### Object detection labelling: [labelImg](https://github.com/tzutalin/labelImg)
check the presentation

#### DDRNet training
\----------------------------------------------------------------  
gathering images and json files  

\----------------------------------------------------------------  
the polygon should be separated, not united as multipolygon

\----------------------------------------------------------------  
crop images and changes the label polygon correspondingly  
3 sizes of images (should be times of 32, resize to 320\*1920): 
+ 320\*1900 (0, 'jpeg')
+ 1080\*1920 (1, 'image', validated: 380~700\*1920)
+ 360\*1900 (-1, 'camera, jpg', validated: 20~340\*1900)  

designated size: 320\*1920  

\----------------------------------------------------------------  
replace class 'on_rails' with 'train'  

\----------------------------------------------------------------  
generate id images for training (json2labelImg.py in cityscapesScripts/preparation)  

\----------------------------------------------------------------  
generate train list  

\----------------------------------------------------------------  
parameters mentioned in the paper:  
![cityscapes](https://github.com/MingZx8/USAnnotation/blob/main/example/CityScapes.png)  
![camvid](https://github.com/MingZx8/USAnnotation/blob/main/example/CamVid.png)  

\----------------------------------------------------------------  
configure  

\----------------------------------------------------------------  
```
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23.yaml
```
```
CUDA_AVAILABLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/cityscapes/ddrnet23.yaml
```

### Sep 2021
#### quality check
##### Semantic segmentation
|       | 2020 | 2021 Spring | 2021 Summer | Total |
| ----- | ---- | ----------- | ----------- | ----- |
| Junwon | `102` | `100` | `104` | `306` |
| James | `30`/104 | `7`/118 | `105`/105 | `152`/327 |
| Yazan | 104 | 100 | 104 | 308 |

|       | Truck | 
| ----- | ---- | 
| Junwon | `74` | 
| James | `73` |
| Yazan | 72 |

##### Object Detection
|       | Truck | 
| ----- | ---- | 
| Junwon | `644`/1000 | 
| James | `983` |
| Yazan | 990 |


### Oct 2021
#### Object detection
1543 valid images.  
seed(12)   
1232/311 -> train/val.  

##### yolo4x-mish (640*\640)
1. [pre-trained weights:](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)
2. edit '.cfg':
3. change line batch to `batch=64`
4. change line subdivisions to `subdivisions=16` (depends on the GPU capacity)
5. change line `max_batches` to (classes*2000, but not less than number of training images and not less than 6000), f.e. max_batches=6000 if you train for 3 classes: 10 \* 2000
6. change line steps to 80% and 90% of max_batches, f.e. `steps=4800,5400`
7. set network size width=416 height=416 or any value multiple of 32
8. change line classes=80 to your number of objects in each of 3 [yolo]-layers
9. change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.
10. ~~when using [Gaussian_yolo] layers, change [filters=57] filters=(classes + 9)x3 in the 3 [convolutional] before each [Gaussian_yolo] layer~~
11. Create file obj.names in the directory build\darknet\x64\data\, with objects names - each in new line
12. Create file obj.data in the directory build\darknet\x64\data\, containing (where classes = number of objects)```classes = 2  
train  = data/train.txt  
valid  = data/test.txt  
names = data/obj.names  
backup = backup/```

##### imporvement
1. for training for small objects (smaller than 16x16 after the image is resized to 416x416) - set layers = 23 instead of L895 (L1046 in mish)
2. set stride=4 instead of L892 (L1043)
3. set stride=4 instead of L989 (L1184)
4. 


