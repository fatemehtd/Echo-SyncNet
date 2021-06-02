# Echo-SyncNet:
### A Neural Network for Synchronization of Cardiac Echo

*Example synchronization of two perpendicular views, AP4 and AP2. 
![](resources/ap4-ap2.gif)
For each view, we show trajectories of the cines in the embedding space. Using a principal component analysis reduction approach, we reduce the dimensionality of the embedding from 128 to 1 for visualization. We examine the synchronization of an AP4 and AP2 echo cine by comparing three distinct cardiac events: the earliest opening of the mitral valve, the maximum contraction of the left ventricle and the earliest closing of the mitral valve.

*Example synchronization of two other cardiac views, AP4 and PLAX, showing the extensibility and generalizability of the proposed model to other cardiac views.
![](resources/plax-ap4.gif)


*Example synchronization of three unique cardiac view angles, AP4, AP2, and PLAX, along with their trajectories in the  embedding space. *
![](resources/ap2ap4plax-sync.gif)


*Example synchronization of four synched cines captured from AP4, AP2, AP5, and PLAX views along with their trajectories in the  embedding space.
![](resources/4view-sync.gif)

**The need for automatic echo synchronization:**
* Calculation of clinical measurements in cardiac echo often require or benefit from having multiple synchronized views or accurately annotated keyframes.
* Traditional methods to synchronize echo rely on external factors such as an electrocardiogram which may not always be available; especially in the point of care setting.

To address these points we propose *Echo-SynNet* a neural network-based framework for automatic synchronization of cardiac echo. Echo-SynNet is trained using only **self-supervised** methods and is hence cheap to train or finetune on any dataset.

<img align="left" src="https://i.imgur.com/kIgSMsO.png" width="350"/> Echo-SyncNet is an encoder style CNN trained to produced low dimensional and feature-rich embedding sequences cardiac ultrasound videos. The embedding vectors carry a powerful semantic understanding of the structure and phase of the heartbeat. Videos can be synchronized simply by performing feature matching on their embedding sequences. 
___
Echo-SyncNet is trained on a dataset of 3070 unannotated echo studies. We use a multiobjective self-supervised loss, described in detail in our paper, to promote the consistency of embedding features across multiple training samples.


## Notebook Demo: Coming Soon!




