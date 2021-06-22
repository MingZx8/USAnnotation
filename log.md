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

### 20-26 June, 2021

