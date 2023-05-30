
# AutoEnsemble: Automated Ensemble Search Framework for Semantic Segmentation Using Image Labels




## Abstract
A key bottleneck of employing state-of-the-art semantic segmentation networks in the real world is the availability of training labels. Standard semantic segmentation networks require massive pixel-wise annotated labels to reach state-of-the-art prediction quality. Hence, several works focus on semantic segmentation networks trained with only image-level annotations. However, when scrutinizing the state-of-the-art results in more detail, we notice that although they are very close to each other on average prediction quality, different approaches perform better in different classes while providing low quality in others. To address this problem, we propose a novel framework, AutoEnsemble, which employs an ensemble of the "pseudo-labels" for a given set of different segmentation techniques on a class-wise level. Pseudo-labels are the pixel-wise predictions of the image-level semantic segmentation frameworks used to train the final segmentation model. Our pseudo-labels seamlessly combine the strong points of multiple segmentation techniques approaches to reach superior prediction quality. We reach up to 2.4% improvement over AutoEnsemble's components. An exhaustive analysis was performed to demonstrate AutoEnsemble's effectiveness over state-of-the-art frameworks for image-level semantic segmentation.

## Overview
![Overall architecture](./figures/ICIP_framework.png)

<br>


## Pseudo-Code
![AutoEnsamble](./figures/Code.png)

<br>

## Complexity Analysis
Let $\{ Comp_1, Comp_2, ..., Comp_N \}$ be the list of Components. Each component takes as input as specific image $i$ from the list of all images $I$ and gives  as output class activation maps $CAM_n^{i,c}$ for all classes $c$ in the dataset.
$$Comp_n(i) =  \sum_{c=0}^C CAM_n^{i,c}, for { } 1 \leq n \leq N$$
As the components  = not further defined by the framework, we can only summarize their complexity as follows:
$$O(Step1) = \sum_{n=0}^N O(Comp_n^{training}(i))O(Comp_n^{inference}(i)) \#epochs_{n} 2\#I$$

# Prerequisite
- Python 3.8, PyTorch 1.7.0, anaconda3
- CUDA 10.1, cuDNN 7.6.5


