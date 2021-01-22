# Echo-SyncNet
### A Self Supervised Neural Network for Synchronization and Cardiac Phase Detection in Echo

Example synchronization of three unique cardiac view angles.
![](resources/ap2ap4plax-sync.gif)

<img align="right" src="https://i.imgur.com/kIgSMsO.png" width="600"> Echo-Synet employs 2D and 3D convolutional neural network layers to encode cardiac echo into low dimensional and feature rich embedding sequences. The embedding vectors carry a powerful semantic undertsanding of the structure and phase of the heart beat, hence multiple videos can be aligned simply by performing feature matching of their embeding sequences. 
Echo-Synet trains on pairs of cardiac echo loops. The encoder network consists of two main modules, a per frame 2D CNN followed by temporal stacking of nearby frames and subsequent 3D convolutional layers.

___

