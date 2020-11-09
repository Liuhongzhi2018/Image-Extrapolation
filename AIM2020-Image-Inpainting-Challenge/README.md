# AIM2020 Challenge on Extreme Image Inpainting

### Data access

By accessing the data the participants implicitly agree with the terms and conditions of the challenge.

 
### Data overview

We are using a partition of ADE20k dataset with a large diversity of contents. For this track only the images are to be used. The use of any other additional information (semantics, object instances) is not allowed.

The dataset is divided into:

    train data
    validation data
    test data

We provide the name of the images of the ADE20k's training set that are to be used for training. For validation a subset of the ADE20k's validation set will be used. The testing set will be comprised of both images from ADE20k and additional sources.

### Masks

For this competition we are using three different types of masks:

- Rectangular masks with width and height between 30-70% of each dimension

- Randomly drawn brush strokes as used in generative inpainting repository

- Our own method for generating masks based on cellural automata (see the masks.py)

### Download

Please use the following link to find the names of the 10330 images to be used for training under the text file name new_image_names.txt. A mask generation script in python can be found in the file masks.py.

You can optionally use the script to download and decompress the ADE20k dataset we provide.


## AIM Workshop and Challenges @ ECCV 2020 - Image Inpainting Challenge

### Important dates

    2020.05.08 Release of train data (input and output images) and validation data (only input)
    2020.05.15 Validation server online
    2020.07.03 Final test data release (only input images)
    2020.07.10 Test output results submission deadline
    2020.07.10 Fact sheets and code/executable submission deadline
    2020.07.12 Preliminary test results release to the participants
    2020.07.22 Paper submission deadline for entries from the challenge
    2020.08.28 AIM workshop and challenges, results and award ceremony (ECCV 2020, Glasgow, UK)

### Challenge overview

The 2nd edition of AIM: Advances in Image Manipulation workshop will be held August 28, 2020 in conjunction with ECCV 2020 in Glasgow, UK.

Image manipulation is a key computer vision tasks, aiming at the restoration of degraded image content, the filling in of missing information, or the needed transformation and/or manipulation to achieve a desired target (with respect to perceptual quality, contents, or performance of apps working on such images). Recent years have witnessed an increased interest from the vision and graphics communities in these fundamental topics of research. Not only has there been a constantly growing flow of related papers, but also substantial progress has been achieved.

Each step forward eases the use of images by people or computers for the fulfillment of further tasks, as image manipulation serves as an important frontend. Not surprisingly then, there is an ever growing range of applications in fields such as surveillance, the automotive industry, electronics, remote sensing, or medical image analysis etc. The emergence and ubiquitous use of mobile and wearable devices offer another fertile ground for additional applications and faster methods.

This workshop aims to provide an overview of the new trends and advances in those areas. Moreover, it will offer an opportunity for academic and industrial attendees to interact and explore collaborations.

Jointly with AIM workshop we have an AIM challenge on examble-based single image inpainting, that is, the task of predicting the values of missing pixels in an image so that the completed result looks realistic and coherent. The challenge uses a modified version of the publicly available ADE20k dataset and has 2 tracks:

- Extreme Image Inpainting

- Extreme Image Inpainting guided by pixel-wise semantic labels

The aim is to obtain a network design / solution capable to produce high quality results with the best perceptual quality and similar to the reference ground truth.

 

The top ranked participants will be awarded and invited to follow the ECCV submission guide for workshops to describe their solution and to submit to the associated AIM workshop at ECCV 2020.

More details are found on the data section of the competition.
Competition

The training data is already made available to the registered participants.
Provided Resources

Scripts: With the dataset the organizers will provide scripts to facilitate the reproducibility of the images and performance evaluation results after the validation server is online. More information is provided on the data page.

Contact: You can use the forum on the data description page (highly recommended!) or directly contact the challenge organizers by email (Evangelos Ntavelis <entavelis [at] ethz.ch>, Siavash Bigdeli <siavash.bigdeli [at] sem.ch>, Andres Romero <roandres [at] ethz.ch> and Radu Timofte <Radu.Timofte [at] vision.ee.ethz.ch>) if you have doubts or any question.


This Challenge was organized by [ETH Zurich - Computer Vision Lab](https://vision.ee.ethz.ch) and  [CSEM SA](https://www.csem.ch/Home) 
