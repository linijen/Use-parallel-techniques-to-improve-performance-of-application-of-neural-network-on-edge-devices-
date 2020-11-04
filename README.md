# Use-parallel-techniques-to-improve-performance-of-application-of-neural-network-on-edge-devices-
## Introduction
As researches on the topic of Deep Learning(DL) become mature, these neural networks are getting much heavier than past. That is to say, lots of computing power is required, including GPU and some parallel algorithm behind it, like distributing training. However, if we take power consumption or GPU utilization into consideration, in most case, many models waste a lot of resources. Thus, how to reach the maximum utilization of hardware accelerators while minimizing waste of energy becomes a vital issue. The central concept behind it is energy efficiency and hardware-aware deployment strategy. Based on these stuffs to extend, it is not only for training but also quite important for inference too. If the model is energy-efficient across platforms, then we could expect that it might have a great performance(latency) on the platforms which are power limited. Therefore, the objective we want to achieve is to use some parallel techniques to efficiently deploy models to a platform, which is not that powerful like server-class workstation, or be equipped with high performance accelerator like Titan, while still keeping its performance. 
## Statement of Problem
If we want to deploy a very deep model to less powerful edge devices to do a real-time object detection demo or other application, we need to overcome some challenges below. First, we need to get a model is efficient enough to be computed in CPU of edge devices, or by using some accelerators, like Edge TPU, to accelerate the forward computing, which would be out of scope of this course, so we don’t spend too much time on this topic. Second, I/O problem would become bottleneck in this scenario. The reason is that, we would use Edge TPU to do forward computing. Thus, if we could not make I/O keep up with inference speed of accelerator, it means we could not fully utilize the advantages of the accelerator. 

Therefore, what we will do is using multithread techniques to maximize the performance, frames per second, in the live demo to its theoretical values, and to make sure that it will not limited by I/O problems. 
## Proposed Approach
We choose HardNet68 as the backbone of SSD object detection model, which would be deployed to Raspberry Pi 4. Main program would launch multiple threads to complete object detection tasks. The overall flow could be separated into 4 stages. First, camera would consume multiple frames, which could be adjusted as hyperparameters according to the requirement. Second, one thread would complete the image preprocessing part, including BGR to RGB, central crop and normalization. Third, coordinate with two distinct Edge TPU accelerators. The last part is to draw the output of the third stage onto original frames to label up where the objects are.  
## KEYWORDS
Hardware-friendly, Energy Efficiency, Edge Devices, Multi-threads, Performance Optimization
Language Selection
We would choose Python to implement this project as first choice because the product of TensorFlow and Edge TPU is more mature and convenient with Python than C++. However, if it is needed to use C++ to get more control of  threads, we will also use it to get better performance.
 
## Related Work
Due to little resource of tutorial about Edge TPU(compared with GPU) and also it is not open-source. We consider that this would be the first work which is focus on optimization for real-time demo. Actually, there are still abundant of code for demo object detection model on Raspberry Pi available. However, little did they, including official tutorials, be focus on the I/O problems, especially when using multiple Edge TPU to demonstrate.

## Statement of Expected Results
We expect using multiple Edge TPU would speed up the frames per second linearly compared with original just one accelerator. However, if we don’t balance the workload of each thread, it may be limited by the capture rate of camera, so that its performance would be poor, under the theoretical value. We also expected that after we balance them, launch the right number of threads, we could achieve maximum of FPS.

## Timetable
11月第一週	環境建置
11月第二週	將模型成功跑在Edge TPU
11月第三週	利用兩根TPU進行加速
11月第四週	優化不同Thread間的workload
12月第一周	嘗試優化I/O，減少不必要function call

Table 1: Timetable to schedule when to complete the project step by step.


## REFERENCES
[1]	https://github.com/Kao1126/EdgeTPU-FaceNet 
[2]	https://coral.ai/docs/reference/edgetpu.detection.engine/


