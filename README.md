# This is the official repository for the paper: [When Will Breakfast be Ready: Temporal Prediction of Food Readiness Using Deep Convolutional Neural Networks on Thermal Videos](https://tars.clarkson.edu/papers/WhenWillBreakfast_ICMEW2018.PDF)
Authors: [Yijun Jiang](https://www.linkedin.com/in/nickjyj), Miao Luo, [Sean Banerjee](https://tars.clarkson.edu/sean), and [Natasha Kholgade Banerjee](https://tars.clarkson.edu/natasha)

---


https://user-images.githubusercontent.com/46984040/121628825-ffead180-ca47-11eb-96fc-c7822ceca75b.mov


# Abstract
In this paper, we perform prediction of food readiness during cooking by using deep convolutional neural networks on thermal video data. Our work treats readiness prediction as ultra-fine recognition of progression in cooking at a per-frame level. We analyze the performance of readiness prediction for eggs, pancakes, and bacon strips using two types of neural networks: a classifier network that bins a frame into one of five classes depending on how far cooking has progressed at that frame, and a regressor network that predicts percentage of cooking time spent at each frame. Our work provides classification accuracies of 98% and higher within one step of the ground truth class using the classifier, and provides an average error of within 20 seconds for the elapsed time predicted using the regressor when compared to ground truth.


---

# Citation
@inproceedings{jiang2018will,
  title={When Will Breakfast be Ready: Temporal Prediction of Food Readiness Using Deep Convolutional Neural Networks on Thermal Videos},
  author={Jiang, Yijun and Luo, Miao and Banerjee, Sean and Banerjee, Natasha Kholgade},
  booktitle={2018 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}
