---
layout: post
title:  "General Contrastive Regression Loss for IR"
date:   2022-10-11 17:11:31 +0900
category: [Research]
tags: [Deep Learning, Computer Vision, Knowledge Distillation, IR, Unfinished]
header-includes:
   - \usepackage{bbm}
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Abstract
Appearance-based gaze estimation has been very successful with the use of deep learning. And there have been many works that improved domain generalization for gaze estimation. However, even though there have been much progress in domain generalization for gaze estimation, most of the recent work have been focused on cross-dataset performance-- accounting for different distributions in illuminations, head pose, and lighting. Although improving gaze estimation in different distributions with RGB images is important, near infrared image based gaze estimation is also critical for gaze estimation in the dark. Also there are inherent limitations relying solely on supervised learning for regression tasks. This paper contributes to solving these problems and proposes GazeCWL, a novel framework for gaze estimation with near-infrared images using contrastive learning. This leverages a adversarial attack technique for data augmentation and a novel contrastive loss function specifically for regression tasks that effectively clusters the features of different samples in the latent space. Our model outperforms previous domain generalization models in infrared image based gaze estimation and with thorough experiments in ablation studies we prove the efficacy of our method. 


## 1. Introduction
Gaze estimation is a crucial technology that has many critical applications in various domains including human-computer interaction[2], virtual reality[3] and medical analysis[4]. Appearance based gaze estimation leveraging deep learning and convolutional neural networks have gained great traction over the past several years[12,13,14]. There have been especially notable progress in appearance based gaze estimation via supervised learning and domain adaptation for cross-dataset performance accounting for different lighting, illuminations, and head poses. However, near-infrared gaze estimation has not gained much traction in the literature. But with broader usage of gaze estimation in edge devices, there is increasing need for gaze estimation in the dark. And infrared images are much more reliable in these settings. 

Even though domain generalization using more datasets have made improvements in performance, gaze annotations are difficult to obtain. Thus it is very challenging to create large datasets that are representative of real world distributions[16]. 

Self-supervised learning (SSL) has been a successful alternative that hsa gained great traction recently, reducing the reliance on labeled samples[17,18,19]. 

Also training with a diverse group of large datasets alone could be insufficient since each dataset has different distribution in illuminations and poses, but does not represent many of the real world scenarios that can be encountered with edge devices[21].

Contrastive learning has been used in many SSL training frameworks, however, the general contrastive loss is most suited for classification tasks[10]. With regression, the slightest changes in the latent vector representation of features could hinder the performance drastically. Thus using the similarity between labels can be exploited as a proxy for the distance between samples in the feature space. 

To the best of our knowledge, we are the first to improve gaze estimation with near infrared images using appearance based deep learning and contrastive learning. 

We propose the following:
1. GazeCWL, a novel framework for gaze estimation with near infrared images. GazeCWL works as a teacher-student model and leverages two novel techniques along with the AI Hub dataset[15] for knowledge distillation. 
2. Data augmentation methods for gaze estimation in near-infrared images. Using both adversarial attack and data augmentation suited for near infrared images for more effective adaptation. 
2. Propose CRWL, a novel contrastive loss function for better clustering of features in the latent space. 

Our model achieves significant performance improvements compared to existing domain generalization methods. 


## 2. Related Works
### 2.1. Appearance based gaze estimation
Appearance based gaze estimation has recently gained great attention as it does not require dedicated devices[6]. Appearance based methods directly map images to the gaze vector and performs better than model methods which rely eye-geometry[12,24]. 

### 2.2. Domain adaptation
Several datasets are frequently used for training gaze estimation models, including ETH-XGaze[8], Gaze Capture[5], and MPII[7]. However as each of these datasets have different distributions in lighting, illumination, and pose, several methods have been previously proposed to increase domain generalization and thus improve corss-dataset performance. 

Many works have employed adversarial training for better domain generalization. And others have used data augmentation techniques, and better feature alignment techniques. 

### 2.3. Contrastive learning
Contrastive learning is a form of unsupervised learning where the goal is to increase the distance between the anchor sample and negative samples and decrease the distance between the anchor sample and positive samples. 

The most notable version of contrastive loss is NT-Xent, highlighted by [9].

Previously contrastive loss has been used in the context of classification tasks and is thus not best suited for regression tasks [10]. 

Wang et al.[10] proposed a contrastive loss by using the distribution of the labels as the weights with KL-divergence. 

Jindal et al.[20] recently proposed a GazeCRL, a contrastive learning framework for gaze estimation. 

[//]: # Normalized Temperature-scaled Cross Entropy Loss  
[//]: # $$ \mathbb{l}_{i,j} = -\log\frac{\exp\left(\text{sim}\left(\mathbf{z}_{i}, \mathbf{z}_{j}\right)/\tau\right)}{\sum^{2N}_{k=1}\mathcal{1}_{[k\neq{i}]}\exp\left(\text{sim}\left(\mathbf{z}_{i}, \mathbf{z}_{k}\right)/\tau\right)} $$  

## 3. Methodology
Our objective is to improve the performance of gaze estimation with near infrared images. Since gaze is a very subtle feature from external appearance, we employ data augmentation and contrastive learning to better cluster features in the latent space yields better performance. 

$$ \min_{f}E_{x^{\tau},y^{\tau}}[L(f(x^{\tau}),g^{\tau})] $$

### 3.1. Overview
We use the backbone of our gaze estimation model following EyeTracker[5] With pretrained weights trained on MPII and XGaze. Itracker takes three normalized images as inputs: left eye, right eye, face. The output of the model is the 2-dimensional camera plane. 

Teacher model and student model.  
$$ x = i(g(x^{rgb})), i(x^{ir}) $$

Contrastive loss weighted with label. 
Together with supervised learning with outputs of the teacher model as the pseudo label. 

Backpropagate to train the feature extractor with the contrastive loss. And train the predictor with the supervised output of the model. 

Optimized for validation with l2 distance.
On the 2-dimensional camera plane. Again, using the outputs of the teacher model as the pseudo label.    
$$ \lvert\lvert \hat{y}_{t} - \hat{y}_{s} \rvert\rvert $$

### 3.2. Adversarial Attack
We then apply two types of data augmentation. One on the source model by converting the RGB image into grayscale by only taking the red intensity. And then applying the following adversarial attack on the target domain on both the source domain and paired target domain images. 

FGSM One step scheme[22]  
$$ x^{\prime} = x + \epsilon \cdot sign(\triangledown_{x}L(f(x),y)) $$

projected gradient descent (PGD)[23]  
$$ x^{t+1} = \Pi_{x+S}(x^{t} + \epsilon \cdot sign(\triangledown_{x}L(f(x),y))) $$ 

We use High-Frequency Component[21], which utilizes both adversarial techniques.  
$$ x^{\prime s,t}_{i} = FGSM(x^{s,t}_{i},y^{s,t}_{i},L_{gaze}) || PGD(x^{s,t}_{i},y^{s,t}_{i},L_{gaze})$$

$$
x^{\prime s,t}_{i} = 
    \left\{\begin{matrix}
        FGSM(x^{s,t}_{i},y^{s,t}_{i},L_{gaze}), 0.5\\
        PGD(x^{s,t}_{i},y^{s,t}_{i},L_{gaze}), 0.5
    \end{matrix}\right.
$$
### 3.3. Contrastive regression loss for gaze estimation
Contrastive loss has proved to be a powerful method for unsupervised learning in classification tasks. 
We propose CRWL, a novel contrastive regression loss function for gaze estimation. 
Contrary to classification tasks, distancing method has to be scaled relative to the sample's proximity to the anchor sample inferred by the labels[10]. Also it should be scaled accordingly since the similarities between samples are relatively similar between gaze samples.   

Here is the proposed contrastive loss for regression tasks.  
$$ -\log\frac{\sum_{j} exp(sim(z_{i}, z_{j})/\tau)}{\sum_{k}\mathbb{1}_{k \ne i}\lvert S_{i, k} \rvert \cdot exp(sim(z_{i}, z_{k})/\tau)} $$

Contrast to the NT-Xent, we scale the negative samples by the similarity between the two labels since subtle distances in features is crucial in regression tasks. 

Cosine similarity  
$$ sim(x,y) = \frac{x \cdot y}{\lvert\lvert x \rvert\rvert \lvert\lvert y \rvert\rvert} $$

The similarity function for weighing the negative samples is as follows:
$$ S = -log \frac{sim(g_{i}, g_{k})}{cos(\pi/60)} $$
The additional division has been added to account for the kappa angle - the difference between visual and pupillary axis.

### 3.4. GazeCWL
[//]: Psuedocode
$$ \documentclass{article}
\usepackage{algpseudocode}
\begin{document}
Algorithm Contrastive Regression Loss: CRL\hline
Input: Data, pretrained network, feature extractor, projection head
Output: Trained Network
\begin{algorithmic}
\While not converged do
  Get batch data
  Conduct augmentation
  Feed into Model
  Calculate CRWL
  Calculate L2
  Backpropagate 

\State $i \gets 10$
\If{$i\geq 5$} 
    \State $i \gets i-1$
\Else
    \If{$i\leq 3$}
        \State $i \gets i+2$
    \EndIf
\EndIf 
\end{algorithmic}

\end{document} $$  

Supervised learning loss, Huber loss  
$$
L_{\delta} = 
\frac{1}{2}(y - \hat{y})^{2} if \left | (y - \hat{y})  \right | < \delta
||
\delta ((y - \hat{y}) - \frac1 2 \delta) otherwise
$$

$$
L_{\delta} = 
    \left\{\begin{matrix}
        \frac{1}{2}(y - \hat{y})^{2} & if \left | (y - \hat{y})  \right | < \delta\\
        \delta ((y - \hat{y}) - \frac1 2 \delta) & otherwise
    \end{matrix}\right.
$$

The final loss is follows:  
$$ \gamma L_{\delta} $$  

l2 loss between prediction and output. Use teacher output as the pseudo label. 

[//]: - Domain Discriminator + Attention in Feature Map Latent Vector

## 4. Experiments
### 4.1. Setup
Pretrain the teacher with Eth-XGaze and MPII Dataset. Additional training and validation on near infrared images of AI Hub dataset[15].  
### 4.2. Training Details
We perform our experiments on NVIDIA 3080 GPU. The resolution for input images in all the experiments is set as 224×224, following the conventions of [5], while different from [25], which employs 448×448as the resolution of input for training on Gaze360. We take ResNet-50 [26] as the backbone to extract features for all experiments  if  without  extra  annotation,  a  2-layers  MLP  as CR predictor to generate 128-dim CR feature vectors, and an FC layer to regress a 2-dim gaze vector for pitch and yaw angles respectively.   For domain generalization task CDGon source domainDE, we follow [27], set the batch size as 16, use the Adam optimizer with a learning rate of 1×10−3 and train for 100 epochs using a decay factor 0.1 every 10 epochs. For CDG on source domainDG, we follow [28], set the batch size as 16, use the Adam optimizer with a learning rate of 1×10−3 and train for 100 epochs.  For domain adaptation tasks, the hyperparameter setting keeps the same as that in domain generalization task CDG in source domainDE.
### 4.3. Domain adaptation
$$ \begin{center}  
 \begin{tabular}{||c c c c||} 
 \hline
 Method & IR Performance \\ [0.5ex] 
 \hline\hline
 Baseline[11] & 2.356 \\ 
 
 CRGA & 2.196 \\
 
 GazeCWL & 1.823 \\
 
 [1ex] 
 
\end{tabular}
\end{center} $$

### 4.4. Ablation Study
#### 4.4.1 Hyperparameter Tuning  
$$ \begin{center}  
 \begin{tabular}{||c c c c||} 
 \hline
 Hyperparameter & Performance \\ [0.5ex] 
 \hline\hline
 \phi=0.1 & 2.245 \\ 
 
 \phi=0.5 & 2.078 \\
 
 \phi=1.0 & 1.823 \\
 
 \phi=10 & 2.178 \\
 
  [1ex] 
 
\end{tabular}
\end{center} $$
#### 4.4.2. Loss function comparison  
$$ \begin{center}  
 \begin{tabular}{||c c c c||} 
 \hline
 Method & & IR \\ [0.5ex] 
 \hline\hline
 l2 & 2.547 \\ 
 
 huber & 2.287 \\
 
 l2+crwl & 1.936 \\
 
 huber+crwl & 1.823 \\
[1ex] 
 
\end{tabular}
\end{center} $$

#### 4.4.3. Epoch  
![Epoch](/assets/posts/research/3.IR_CRL/Ablation_study_on_iterations_of_self-training.png "Epoch")  

#### 4.4.4. Similarity Function
$$ \begin{center}  
 \begin{tabular}{||c c c c||} 
 \hline
 Method & & IR \\ [0.5ex] 
 \hline\hline
 l2 & 2.017 \\ 
 
 cosine similarity & 1.947 \\
 
 cosine+modification & 1.823 \\
[1ex] 
 
\end{tabular}
\end{center} $$

## 5. Conclusion
In this paper we introduced GazeCWL, a novel contrastive learning framework for gaze estimation with near infrared images. Our framework employs data augmentation techniques specifically for near infrared images and utilize a novel contrastive loss function effective for regression tasks. Furthermore, we showed that GazeCWL is effective at clustering relevant features closer together in the latent space.GazeCWL can be used as a general framework for training with near infrared images on regression tasks, thus can be explored in the future for applications.

### References
[1]  Philip Bachman, R Devon Hjelm, and William Buchwalter.Learning representations by maximizing mutual informationacross views.Advances in Neural Information ProcessingSystems, 32:15535–15545, 2019.  
[2] X. Zhang, Y. Sugano, and A. Bulling, “Evaluation of appearance-based   methods and implications for gaze-based applications,” in Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, ser. CHI ’19. New York, NY,  SA: Association for Computing Machinery, 2019.  
[3] Alisa Burova, John M ̈akel ̈a, Jaakko Hakulinen, Tuuli Keski-nen, Hanna Heinonen, Sanni Siltanen, and Markku Turunen.Utilizing vr and gaze tracking to develop ar solutions for in-dustrial maintenance.  InProceedings of the 2020 CHI Con-ference on Human Factors in Computing Systems, pages 1–13, 2020.  
[4] Nora  Castner,  Thomas  C  Kuebler,  Katharina  Scheiter,  Ju-liane Richter, Th ́er ́ese Eder, Fabian H ̈uttig, Constanze Keu-tel, and Enkelejda Kasneci.  Deep semantic gaze embeddingand scanpath comparison for expertise classification duringopt viewing.  InACM Symposium on Eye Tracking Researchand Applications, pages 1–10, 2020   
[5] Krafka, Kyle, et al. "Eye tracking for everyone." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.  
[6] Cheng, Yihua, et al. "Appearance-based gaze estimation with deep learning: A review and benchmark." arXiv preprint arXiv:2104.12668 (2021).  
[7] Zhang, Xucong, et al. "Mpiigaze: Real-world dataset and deep appearance-based gaze estimation." IEEE transactions on pattern analysis and machine intelligence 41.1 (2017): 162-175.  
[8] Zhang, Xucong, et al. "Eth-xgaze: A large scale dataset for gaze estimation under extreme head pose and gaze variation." European Conference on Computer Vision. Springer, Cham, 2020.  
[9] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.  
[10] Y. Wang et al., "Contrastive Regression for Domain Adaptation on Gaze Estimation," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 19354-19363, doi: 10.1109/CVPR52688.2022.01877.  
[11] Xucong Zhang, Yusuke Sugano, and Andreas Bulling.  Re-visiting data normalization for appearance-based gaze esti-mation. InProceedings of the 2018 ACM Symposium on EyeTracking Research & Applications, pages 1–9, 2018  
[12] X. Zhang,  Y. Sugano,  M. Fritz,  and A. Bulling,  “Appearance-based gaze estimation in the wild,”  inProceedings of the IEEE conference on computer vision and pattern recognition, pp. 4511–4520, 2015.  
[13] X. Zhang, Y. Sugano, M. Fritz, and A. Bulling, “It’s written all over your face: Full-face appearance-basedgaze estimation,” inProceedings of the IEEE Conference on Computer Vision and Pattern RecognitionWorkshops, pp. 51–60, 2017.  
[14] K. Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik, and A. Torralba, “Eyetracking for everyone,” inProceedings of the IEEE conference on computer vision and pattern recognition,pp. 2176–2184, 2016.
[15] Song. "Eye Movement Video Data" AIHub, 2022. Web1. 1 Sep 2022. <https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=548>  
[16] S. Ghosh, A. Dhall, M. Hayat, J. Knibbe, and Q. Ji, “Automatic gaze analysis: A survey of deep learningbased approaches,”arXiv preprint arXiv:2108.05479, 2021.  
[17] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representationlearning,”  inProceedings  of  the  IEEE/CVF  conference  on  computer  vision  and  pattern  recognition,pp. 9729–9738, 2020.  
[18] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visualrepresentations,” inInternational conference on machine learning, pp. 1597–1607, PMLR, 2020.  
[19] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires,Z. Guo, M. Gheshlaghi Azar,et al., “Bootstrap your own latent-a new approach to self-supervised learning,”Advances in Neural Information Processing Systems, vol. 33, pp. 21271–21284, 2020.  
[20] Jindal, Swati, and Roberto Manduchi. "Contrastive Representation Learning for Gaze Estimation." arXiv preprint arXiv:2210.13404 (2022).  
[21] Liu, Ruicong, et al. "Jitter Does Matter: Adapting Gaze Estimation to New Domains." arXiv preprint arXiv:2210.02082 (2022).  
[22] Goodfellow, I. J.; Shlens, J.; and Szegedy, C. 2014. Explain-ing  and  harnessing  adversarial  examples.arXiv  preprintarXiv:1412.6572.  
[23] Madry,  A.;  Makelov,  A.;  Schmidt,  L.;  Tsipras,  D.;  andVladu, A. 2017a.   Towards deep learning models resistantto adversarial attacks.arXiv preprint arXiv:1706.06083.  
[24] D. W. Hansen and Q. Ji, “In the eye of the beholder:  A survey of models for eyes and gaze,”IEEEtransactions on pattern analysis and machine intelligence, vol. 32, no. 3, pp. 478–500, 2009.  
[25] Yunfei Liu, Ruicong Liu, Haofei Wang, and Feng Lu.  Gen-eralizing  gaze  estimation  with  outlier-guided  collaborativeadaptation.   InProceedings of the IEEE/CVF InternationalConference on Computer Vision, pages 3835–3844, 2021.  
[26] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition.   InProceed-ings of the IEEE conference on computer vision and patternrecognition, pages 770–778, 2016  
[27] Xucong   Zhang,   Seonwook   Park,   Thabo   Beeler,   DerekBradley, Siyu Tang, and Otmar Hilliges.  Eth-xgaze: A largescale dataset for gaze estimation under extreme head poseand gaze variation.   InEuropean Conference on ComputerVision, pages 365–381. Springer, 2020.  
[28] Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Ma-tusik,  and Antonio Torralba.   Gaze360:  Physically uncon-strained gaze estimation in the wild.  InProceedings of theIEEE/CVF International Conference on Computer Vision,pages 6912–6921, 2019.