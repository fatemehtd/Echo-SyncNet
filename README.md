# Echo-SyncNet
### A Self Supervised Neural Network for Synchronization and Cardiac Phase Detection in Echo

Example synchronization of three unique cardiac view angles.
![](resources/ap2ap4plax-sync.gif)

<img align="left" src="https://i.imgur.com/kIgSMsO.png" width="350"/> Echo-SyncNet is an enocder style CNN trained to produced low dimensional and feature rich embedding sequences cardiac ultrasound videos. The embedding vectors carry a powerful semantic undertsanding of the structure and phase of the heart beat. Vidoes can be synchronized simply by performing feature matching on their embeding sequences. 
___
Echo-SyncNet is trained on a dataset of 3070 unannotated echo studies. We use a multiobjective self supervised loss, described in detail in our paper, to promote the consistency of embedding features across multiple tranining samples.

## Notebook Demo: Coming Soon!




