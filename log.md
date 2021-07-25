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

### July, 2021
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

#### Object detection labelling: [labelImg](https://github.com/tzutalin/labelImg)
check the presentation

#### DDRNet training
1. gathering images and json files
2. crop images and changes the bbox correspondingly
  1. 3 sizes of images: 320\*1900, 1080\*1920 (validated: 380~700\*1920), 360\*1900 (validated: 20~340\*1900)
3. 
