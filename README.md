# Echo-SyncNet:
### A Self Supervised Neural Network for Synchronization and Phase Detection in Cardiac Echo

**The need for automatic echo synchronization:**
* Calculation of clinical measurements in cardiac echo often require or benefit from having multiple synchronized views or accurately annotated keyframes.
* Traditional methods to synchronize echo rely on external factors such as an electrocardiogram which may not always be available; especially in the point of care setting.

To address these points we propose *Echo-SynNet* a neural network-based framework for automatic synchronization of cardiac echo. Echo-SynNet is trained using only self-supervised methods and is hence cheap to train or finetune on any dataset.

*Example synchronization of three unique cardiac view angles.*
![](resources/ap2ap4plax-sync.gif)

<img align="left" src="https://i.imgur.com/kIgSMsO.png" width="350"/> Echo-SyncNet is an encoder style CNN trained to produced low dimensional and feature-rich embedding sequences cardiac ultrasound videos. The embedding vectors carry a powerful semantic understanding of the structure and phase of the heartbeat. Videos can be synchronized simply by performing feature matching on their embedding sequences. 
___
Echo-SyncNet is trained on a dataset of 3070 unannotated echo studies. We use a multiobjective self-supervised loss, described in detail in our paper, to promote the consistency of embedding features across multiple training samples.

## Notebook Demo: Coming Soon!




