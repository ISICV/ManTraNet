# **ManTraNet**: Manipulation Tracing Network For Detection And Localization of Image ForgeriesWith Anomalous Features
<img src="https://www.isi.edu/images/isi-logo.jpg" width="300"/> <img src="http://cvpr2019.thecvf.com/images/CVPRLogo.png" width="300"/> 

***
This is the official repo for the ManTraNet (CVPR2019). For method details, please refer to 

```
  @inproceedings{Wu2019ManTraNet,
      title={ManTra-Net: Manipulation Tracing Network For Detection And Localization of Image ForgeriesWith Anomalous Features},
      author={Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan},
      journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019}
  }
```

***

# Overview
ManTraNet is an end-to-end image forgery detection and localization solution, which means it takes a testing image as input, and predicts pixel-level forgery likelihood map as output. Comparing to existing methods, the proposed ManTraNet has the following advantages:

  1. **Simplicity**: ManTraNet needs no extra pre- and/or post-processing
  2. **Fast**: ManTraNet puts all computations in a single network, and accepts an image of arbitrary size. 
  3. **Robustness**: ManTraNet does not rely on working assumptions other than *the local manipulation assumption*, i.e. some region in a testing image is modified differently from the rest. 

<img src="https://github.com/ISICV/ManTraNet/blob/master/data/result0.png" width="400"/> <img src="https://github.com/ISICV/ManTraNet/blob/master/data/result1.png" width="400"/> 

Technically speaking, ManTraNet is composed of two sub-networks as shown below:
  1. Image Manipulation Trace Feature Extractor: the feature extraction network for the image manipulation classification task, which is sensitive to different manipulation types, and encodes the image manipulation in a patch into a fixed dimension feature vector.
  2. Local Anomaly Detection Network: the anomaly detection network to compare a local feature against the dominant feature averaged from a local region, whose activation depends on how far a local feature deviates from the reference feature instead of the absolute value of a local feature.  

![ManTraNet](https://github.com/ISICV/ManTraNet/blob/master/data/ManTraNet-overview.png)

# Extension
ManTraNet is pretrained with all synthetic data. To prevent overfitting, we 
1. Pretrain the Image Manipulation Classification ([385 classes](https://github.com/ISICV/ManTraNet/blob/master/data/IMC385.png)) task to obtain the Image *Manipulation Trace Feature Extractor*
2. Train ManTraNet with four types of synthetic data, i.e. copy-move, splicing, removal, and enhancement

To extend the provided ManTraNet, one may introduce the new manipulation either to the IMC pretrain task, or to the end-to-end ManTraNet task, or both. It is also worth noting that the IMC task can be a self-supervised task. 

# Dependency
ManTraNet is written in Keras with the TensorFlow backend.
  
  - Keras: 2.2.0
  - TensorFlow: 1.8.0
  
Other versions might also work, but not tested.

# Demo
One may simply download the repo and play with the provided ipython notebook. 

Alternatively, one may play with the inference code using [this google colab link](https://colab.research.google.com/drive/1ai4kVlI6w9rREqqYnTfpk3gM3YX9k-Ek).

# Contact
For any paper related questions, please contact `rex.yue.wu(AT)gmail.com`

# Licence
The Software is made available for academic or non-commercial purposes only. The license is for a copy of the program for an unlimited term. Individuals requesting a license for commercial use must pay for a commercial license.

    USC Stevens Institute for Innovation 
    University of Southern California 
    1150 S. Olive Street, Suite 2300 
    Los Angeles, CA 90115, USA 
    ATTN: Accounting 
  
DISCLAIMER. USC MAKES NO EXPRESS OR IMPLIED WARRANTIES, EITHER IN FACT OR BY OPERATION OF LAW, BY STATUTE OR OTHERWISE, AND USC SPECIFICALLY AND EXPRESSLY DISCLAIMS ANY EXPRESS OR IMPLIED WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, VALIDITY OF THE SOFTWARE OR ANY OTHER INTELLECTUAL PROPERTY RIGHTS OR NON-INFRINGEMENT OF THE INTELLECTUAL PROPERTY OR OTHER RIGHTS OF ANY THIRD PARTY. SOFTWARE IS MADE AVAILABLE AS-IS. LIMITATION OF LIABILITY. TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL USC BE LIABLE TO ANY USER OF THIS CODE FOR ANY INCIDENTAL, CONSEQUENTIAL, EXEMPLARY OR PUNITIVE DAMAGES OF ANY KIND, LOST GOODWILL, LOST PROFITS, LOST BUSINESS AND/OR ANY INDIRECT ECONOMIC DAMAGES WHATSOEVER, REGARDLESS OF WHETHER SUCH DAMAGES ARISE FROM CLAIMS BASED UPON CONTRACT, NEGLIGENCE, TORT (INCLUDING STRICT LIABILITY OR OTHER LEGAL THEORY), A BREACH OF ANY WARRANTY OR TERM OF THIS AGREEMENT, AND REGARDLESS OF WHETHER USC WAS ADVISED OR HAD REASON TO KNOW OF THE POSSIBILITY OF INCURRING SUCH DAMAGES IN ADVANCE.

For commercial license pricing and annual commercial update and support pricing, please contact:

    Rakesh Pandit USC Stevens Institute for Innovation 
    University of Southern California 
    1150 S. Olive Street, Suite 2300
    Los Angeles, CA 90115, USA 

    Tel: +1 213-821-3552
    Fax: +1 213-821-5001 
    Email: rakeshvp@usc.edu and ccto: accounting@stevens.usc.edu

