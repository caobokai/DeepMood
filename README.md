# Multi-view Sequence Prediction

The task of inferring mood disturbance from mobile phone typing dynamics metadata is formulated as a multi-view sequence prediction problem. We develop a deep learning architecture for mood detection using the collected features about alphanumeric characters, special characters, and accelerometer values. Specifically, it is an end-to-end approach based on late fusion to modeling the multi-view time series data. In the first stage, each view of the time series is separately modeled by a recurrent network. The multi-view information is then fused in the second stage through three alternative layers that concatenate and explore interactions across the output vectors from each view.

Reference
---------
Bokai Cao, Lei Zheng, Chenwei Zhang, Philip S. Yu, Andrea Piscitello, John Zulueta, Olu Ajilore, Kelly Ryan and Alex D. Leow. [DeepMood: Modeling Mobile Phone Typing Dynamics for Mood Detection](https://www.cs.uic.edu/~bcao1/doc/kdd17a.pdf). In KDD 2017.
