# Image-Extrapolation
What I cannot create, I do not understand.

This repository is a paper list of image extrapolation and inpainting inspired by @1900zyh's repository [Awsome-Image-Inpainting](https://github.com/1900zyh/Awsome-Image-Inpainting) and @geekyutao's repository [Image Inpainting](https://github.com/geekyutao/Image-Inpainting). As a conditional generation task, I learn from @xiaweihao's repository [awesome-image-translation](https://github.com/xiaweihao/awesome-image-translation).

## Deep-Learning-based Extrapolation Methods
Year|Proceeding|Title|Comment
--|:--:|:--:|:--
2020|WACV 2020| Multimodal Image Outpainting with Regularized Normalized Diversification [[paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Zhang_Multimodal_Image_Outpainting_With_Regularized_Normalized_Diversification_WACV_2020_paper.pdf) [[code]](https://github.com/owenzlz/DiverseOutpaint)|GAN-based
2019|ICCV 2019|Boundless: Generative Adversarial Networks for Image Extension [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Teterwak_Boundless_Generative_Adversarial_Networks_for_Image_Extension_ICCV_2019_paper.pdf) [[code]](https://github.com/recong/Boundless-in-Pytorch)|
2019|ICCV 2019|Very Long Natural Scenery Image Prediction by Outpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Very_Long_Natural_Scenery_Image_Prediction_by_Outpainting_ICCV_2019_paper.pdf) [[code]](https://github.com/z-x-yang/NS-Outpainting)|LSTM
2019|ICCV 2019|COCO-GAN: Generation by Parts via Conditional Coordinating [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_COCO-GAN_Generation_by_Parts_via_Conditional_Coordinating_ICCV_2019_paper.pdf) [[official code]](https://github.com/hubert0527/COCO-GAN) [[unofficial code]](https://github.com/shaanrockz/COCO-GAN)|
2019|SIGGRAPH 2019|Structure-Aware Image Expansion with Global Attention [[paper]](https://dl.acm.org/doi/abs/10.1145/3355088.3365161)|
2019|CVPR 2019|Wide-Context Semantic Image Extrapolation [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Wide-Context_Semantic_Image_Extrapolation_CVPR_2019_paper.pdf) [[code]](https://github.com/shepnerd/outpainting_srn)|
2018|ECCV 2018|Unsupervised Holistic Image Generation from Key Local Patches [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Donghoon_Lee_Unsupervised_holistic_image_ECCV_2018_paper.pdf) [[code]](https://github.com/hellbell/KeyPatchGan)|GAN-based

## Traditional Extrapolation Methods
Year|Proceeding|Title|Comment
--|:--:|:--:|:--
2003|ACM TOG 2003|Poisson Image Editing [[paper]](http://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)|
2007|ACM TOG 2007|Seam Carving for Content-Aware Image Resizing [[paper]](https://www.win.tue.nl/~wstahw/edu/2IV05/seamcarving.pdf)|
2008|CVPRW 2008|Creating and Exploring a Large Photorealistic Virtual Space  [[paper]](http://people.csail.mit.edu/biliana/papers/pieee2009/pieee2009.pdf) [[paper]](https://www.di.ens.fr/~josef/publications/sivic08a.pdf)|
2009|ACM TOG 2009| PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing [[paper]](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)|
2012|SIGGRAPH 2012| Quality Prediction for Image Completion [[paper]](http://johanneskopf.de/publications/completion/completion.pdf)|
2013|CVPR 2013| FrameBreak: Dramatic Image Extrapolation by Guided Shift-Maps [[paper]](http://openaccess.thecvf.com/content_cvpr_2013/papers/Zhang_FrameBreak_Dramatic_Image_2013_CVPR_paper.pdf) [[code]](https://github.com/parvparkhiya/Dramatic_Image_Extrapolation_Implementation/)|
2014|ECCV 2014| Photo uncrop [[paper]](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10599-4_2.pdf) [[code]](https://github.com/abhinavs95/photo-uncrop)|
2014|ACM TOG 2014| BiggerPicture: data-driven image extrapolation using graph matching [[paper]](http://orca.cf.ac.uk/67868/1/biggerpicture_compressed.pdf)|



## Inpainting Deep-Learning-based Methods
Year|Proceeding|Title|Comment
--|:--:|:--:|:--
2012|NIPS 2012| Image denoising and inpainting with deep neural networks [[pdf]](http://papers.nips.cc/paper/4686-image-denoising-and-inpainting-with-deep-neural-networks.pdf)|
2014|GCPR 2014|Mask-specific inpainting with deep neural networks [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-11752-2_43)|
2014|NIPS 2014|Deep Convolutional Neural Network for Image Deconvolution [[pdf]](http://papers.nips.cc/paper/5485-deep-convolutional-neural-network-for-image-deconvolution.pdf)|
2015|NIPS 2015|Shepard Convolutional Neural Networks [[pdf]](https://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf) [[code]](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/Shepard_CNN)|
2015|PR 2015|Application of GRNN neural network in non-texture image inpainting and restoration [[paper]](https://pdf.sciencedirectassets.com/271524/1-s2.0-S0167865515X00080/1-s2.0-S0167865515001476/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQC69BBg2xQzc5%2Bqx1Mia9vgAd9Z%2BVEQlw2EXwhHtF87uQIhAN%2FyYwXz1OY3ixESne5xmMvPBdWlckfHZIuXUCljXfbjKtACCCwQAhoMMDU5MDAzNTQ2ODY1Igy6x8v2DnL1ERq1pmkqrQIyBOEQ4G6TXvk6nQtiggl9BIDTUmcOlnRbsYlDoF7pKT10tW4mP552hAUwlLHzaCsmi6lZiMeMLfCQ57Tz7Xkm1RRXkbLllGhVyZcnthahAlNCiAgy7z2efhSlk0TzyM8kcC8hL%2BJjv9J%2FyP%2FF2txPbXDfhfDOWYjcIUFEvDymAhGPwAsewuOHSaSSCYTXflybtKzvMtQ0aA60h3qIlyJR1RyQxKraQDFU2jZH%2BKer%2BTtxSRvJvztEgiZhJRL9WYTZiAcnD9D0FQIGRx54fMx8Uw%2FcTmpqgdlzsaxO65ccHsYsIuj2c%2BXqCzCPQLQyWmrFuJeXqhrCP8pBecj0kvYdc45%2Buz4QSFcNqXad6%2ByvlrpMLO0NCGiGbod0KJRU9vmarQXXyn8ya0V3MQXxMMPvmO8FOs4Cj2UbctRf%2FlLRLvODofwIZ%2FnRNPWkGw5iuPGtL%2Bi8VnhR4eO%2BDwRFu4%2F3lsbWSCAAnW%2BIaliFwlJnDWWapgf8%2Fkv6RXcgaOzJM8QeLyxFs5WY%2FOHZIJTRjzovezLsbi7hdJicKigrArSZtA2a%2BuqN5zmeeQu4eKRtA6KdGWoDqYYp1qfTbkfShvrB4OuaaedScD7wH9teePXp%2BfS3DVVarjKdw2KY71PEJxXXHFs6PLGoB%2FD4VUjRWo2i%2Ftt9nZJrsbZh%2BV6Vw9rR42fSvHK8CA2AHg0HBYIiITQuOMzE%2FiNUunNaiZmj%2FwIfRc%2BQxpfiC6YDikuWq%2FlIUAhTMVAAUrFaBL6SX6qQl4U4182lcSzwhhkZDuKktq6Po9Ic0594Zd3KLSLI1FAaTtH3x9jVpoKCXBD2ql1QAHvtLZIt5l2oF9XC8InHI%2FsrR%2FhmoQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191203T113355Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQ7UI3ZGS%2F20191203%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=58a5b3a019c0024eb5290b0efbbfd9ea4e06b6fabd1cc6f5025f6457402de0f4&hash=839556fee11fca230dc7ee3539fda5e3fbd99d3bc01bdbe9719d3fa322f38d38&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167865515001476&tid=spdf-87072018-5504-4c0e-92ef-ebfcc30fdf18&sid=bf32171d5938c4482c9ba4c482e4d6e21e87gxrqa&type=client)|CNN-based
2015|AICCSA 2015|Moving objects detection based on thresholding operations for video surveillance systems [[paper]](https://www.researchgate.net/profile/Omar_El_Harrouss/publication/306064382_Moving_objects_detection_based_on_thresholding_operations_for_video_surveillance_systems/links/58447f7108ae8e63e6271e4e/Moving-objects-detection-based-on-thresholding-operations-for-video-surveillance-systems.pdf)|GAN-based
2016|CVPR 2016|Context Encoders: Feature Learning by Inpainting [[pdf]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf) [[code]](https://github.com/pathak22/context-encoder)|CNN-based
2016|SIGGRAPH 2016|High-resolution multi-scale neural texture synthesis [[pdf]](https://wxs.ca/research/multiscale-neural-synthesis/Snelgrove-multiscale-texture-synthesis.pdf)|
2016|ELCVIA 2016|A block-based background model for moving object detection [[paper]](https://elcvia.cvc.uab.es/article/download/v15-n3-elharrouss/855-pdf-en)|GAN-based
2016|JEI 2016|Moving object detection using a background modeling based on entropy theory and quad-tree decomposit [[paper]](https://www.researchgate.net/profile/Omar_El_Harrouss/publication/311091714_Moving_object_detection_using_a_background_modeling_based_on_entropy_theory_and_quad-tree_decomposition/links/5b9bba0945851574f7c940fb/Moving-object-detection-using-a-background-modeling-based-on-entropy-theory-and-quad-tree-decomposition.pdf)|GAN-based
2017|CVPR 2017|Semantic image inpainting with deep generative models [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf) [[code]](https://github.com/moodoki/semantic_image_inpainting)|
2017|CVPR 2017|High-resolution image inpainting using multi-scale neural patch synthesis [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_High-Resolution_Image_Inpainting_CVPR_2017_paper.pdf) [[code]](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting)|
2017|CVPR 2017|Generative Face Completion [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Generative_Face_Completion_CVPR_2017_paper.pdf) [[code]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)|
2017|CVPR 2017|Joint gap detection and inpainting of line drawings [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sasaki_Joint_Gap_Detection_CVPR_2017_paper.pdf)|CNN-based
2017|SIGGRAPH 2017|Globally and Locally Consistent Image Completion [[pdf]](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) [[code]](https://github.com/satoshiiizuka/siggraph2017_inpainting)|
2017|VC 2017|Blind inpainting using the fully convolutional neural network [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs00371-015-1190-z.pdf)|CNN-based
2017|ICVISP 2017|High-resolution image inpainting through multiple deep networks [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8123592&tag=1)|CNN-based
2017|ICDAR 2017|Scene Text Eraser [[paper]](https://arxiv.org/pdf/1705.02772.pdf)|CNN-based
2017|ICCC 2017|A Deep Network Architecture for Image Inpainting [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8322859)|CNN-based
2018|CVPR 2018|Generative Image Inpainting with Contextual Attention [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Generative_Image_Inpainting_CVPR_2018_paper.pdf) [[code]](https://github.com/JiahuiYu/generative_inpainting)|
2018|CVPR 2018|Natural and Effective Obfuscation by Head Inpainting [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Natural_and_Effective_CVPR_2018_paper.pdf)|
2018|CVPR 2018|Eye In-Painting With Exemplar Generative Adversarial Networks [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dolhansky_Eye_In-Painting_With_CVPR_2018_paper.pdf) [[code]](https://github.com/bdol/exemplar_gans)|
2018|CVPR 2018|UV-GAN: Adversarial Facial UV Map Completion for Pose-invariant Face Recognition [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_UV-GAN_Adversarial_Facial_CVPR_2018_paper.pdf)|
2018|CVPR 2018|Disentangling Structure and Aesthetics for Style-aware Image Completion [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gilbert_Disentangling_Structure_and_CVPR_2018_paper.pdf)|
2018|ECCV 2018|Image Inpainting for Irregular Holes Using Partial Convolutions [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf) [[code]](https://github.com/NVIDIA/partialconv)|
2018|ECCV 2018| Contextual-based Image Inpainting: Infer, Match, and Translate [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yuhang_Song_Contextual_Based_Image_ECCV_2018_paper.pdf)|
2018|ECCV 2018|Shift-Net: Image Inpainting via Deep Feature Rearrangement [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhaoyi_Yan_Shift-Net_Image_Inpainting_ECCV_2018_paper.pdf) [[code]](https://github.com/Zhaoyi-Yan/Shift-Net)|CNN-based
2018|NIPS 2018|Image Inpainting via Generative Multi-column Convolutional Neural Networks [[pdf]](http://papers.nips.cc/paper/7316-image-inpainting-via-generative-multi-column-convolutional-neural-networks.pdf) [[code]](https://github.com/shepnerd/inpainting_gmcnn)|
2018|TOG 2018|Faceshop: Deep sketch-based face image editing [[pdf]](https://arxiv.org/pdf/1804.08972.pdf)|
2018|ACM MM 2018|Structural inpainting [[pdf]](https://arxiv.org/pdf/1803.10348.pdf) |
2018|ACM MM 2018|Semantic Image Inpainting with Progressive Generative Networks [[pdf]](https://dl.acm.org/citation.cfm?id=3240625) [[code]](https://github.com/crashmoon/Progressive-Generative-Networks)|
2018|BMVC 2018|SPG-Net: Segmentation Prediction and Guidance Network for Image Inpainting [[pdf]](https://arxiv.org/pdf/1805.03356.pdf)|
2018|BMVC 2018|MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesi [[pdf]](https://arxiv.org/pdf/1805.01123.pdf) [[code]](https://github.com/HYOJINPARK/MC_GAN) |
2018|ACCV 2018|Face Completion with Semantic Knowledge and Collaborative Adversarial Learning [[pdf]](https://arxiv.org/pdf/1812.03252.pdf)|
2018|ICASSP 2018|Edge-Aware Context Encoder for Image Inpainting [[pdf]](http://mirlab.org/conference_papers/International_Conference/ICASSP%202018/pdfs/0003156.pdf)|
2018|ICPR 2018|Deep Structured Energy-Based Image Inpainting [[pdf]](https://arxiv.org/pdf/1801.07939.pdf) [[code]](https://github.com/cvlab-tohoku/DSEBImageInpainting)|
2018|AISTATS 2019|Probabilistic Semantic Inpainting with Pixel Constrained CNNs [[pdf]](https://arxiv.org/pdf/1810.03728.pdf)|
2018|ICRA 2018|Just-in-time reconstruction: Inpainting sparse maps using single view depth predictors as priors [[paper]](https://arxiv.org/pdf/1805.04239v1.pdf)| CNN-based
2018|SPIC 2018|A deep learning approach to patch-based image inpainting forensics [[paper]](https://www.researchgate.net/profile/Xinshan_Zhu3/publication/325802468_A_deep_learning_approach_to_patch-based_image_inpainting_forensics/links/5d3c5fe892851cd0468c487c/A-deep-learning-approach-to-patch-based-image-inpainting-forensics.pdf)|CNN-based
2018|MS 2018|Semantic object removal with convolutional neural network feature-based inpainting approach [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs00530-018-0585-x.pdf)|CNN-based
2018|NCA 2018|Occluded offline handwritten Chinese character recognition using deep convolutional generative adversarial network and improved GoogLeNet [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs00521-018-3854-x.pdf)|GAN-based
2018|TIP 2018|Perceptual Adversarial Networks forImage-to-Image Transformation [[paper]](https://arxiv.org/pdf/1706.09138.pdf)|GAN-based
2018|GRSL 2018|Inpainting of Remote Sensing SST Images With Deep Convolutional Generative Adversarial Network [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8480867)|GAN-based
2018|PRRS 2018|Preliminary Investigation on Single Remote Sensing Image Inpainting Through a Modified GAN [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8486163)|GAN-based
2018|ICCES 2018|Semantic Image Inpainting Vsing Self-Learning Encoder-Decoder and Adversarial Loss [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8639258)|GAN-based
2018|ICNC-FSKD 2018|Image Inpainting Based on Generative Adversarial Networks [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8686914)|GAN-based
2019|ICRA 2019|Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space [[pdf]](https://arxiv.org/pdf/1809.10239.pdf)|
2019|AAAI 2019|Video Inpainting by Jointly Learning Temporal Structure and Spatial Details [[pdf]](https://arxiv.org/pdf/1806.08482.pdf)| Video
2019|CVPR 2019| Pluralistic Image Completion [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Pluralistic_Image_Completion_CVPR_2019_paper.pdf) [[code]](https://github.com/lyndonzheng/Pluralistic) [[project]](http://www.chuanxiaz.com/publication/pluralistic/?tdsourcetag=s_pctim_aiomsg)| Diversity
2019|CVPR 2019| PEPSI: Fast Image Inpainting with Parallel Decoding Network [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sagong_PEPSI__Fast_Image_Inpainting_With_Parallel_Decoding_Network_CVPR_2019_paper.pdf) [[code]](https://github.com/Forty-lock/PEPSI)|
2019|CVPR 2019| Deep Reinforcement Learning of Volume-guided Progressive View Inpainting for 3D Point Scene Completion from a Single Depth Image [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Han_Deep_Reinforcement_Learning_of_Volume-Guided_Progressive_View_Inpainting_for_3D_CVPR_2019_paper.pdf)|
2019|CVPR 2019|Foreground-aware Image Inpainting [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xiong_Foreground-Aware_Image_Inpainting_CVPR_2019_paper.pdf)  |
2019|CVPR 2019 |Privacy Protection in Street-View Panoramas using Depth and Multi-View Imagery [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Uittenbogaard_Privacy_Protection_in_Street-View_Panoramas_Using_Depth_and_Multi-View_Imagery_CVPR_2019_paper.pdf) |
2019|CVPR 2019|Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Learning_Pyramid-Context_Encoder_Network_for_High-Quality_Image_Inpainting_CVPR_2019_paper.pdf) [[code]](https://github.com/researchmm/PEN-Net-for-Inpainting)|CNN-based
2019|CVPR 2019|Blind Visual Motif Removal from a Single Image [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hertz_Blind_Visual_Motif_Removal_From_a_Single_Image_CVPR_2019_paper.pdf)|CNN-based
2019|CVPR 2019|Deep Flow-Guided Video Inpainting [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Flow-Guided_Video_Inpainting_CVPR_2019_paper.pdf) [[project]](https://nbei.github.io/video-inpainting.html)| Video
2019|CVPR 2019|Deep video inpainting [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Video_Inpainting_CVPR_2019_paper.pdf)|Video
2019|CVPR Workshop 2019 |VORNet: Spatio-temporally Consistent Video Inpainting for Object Removal [[pdf]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Chang_VORNet_Spatio-Temporally_Consistent_Video_Inpainting_for_Object_Removal_CVPRW_2019_paper.pdf)| Video
2019|TNNLS 2019|PEPSI++: Fast and Lightweight Network for Image Inpainting [[pdf]](https://arxiv.org/pdf/1905.09010.pdf)|GAN-based
2019|IJCAI 2019|MUSICAL: Multi-Scale Image Contextual Attention Learning for Inpainting [[pdf]](https://www.ijcai.org/proceedings/2019/0520.pdf) |
2019|IJCAI 2019|Generative Image Inpainting with Submanifold Alignment [[pdf]](https://www.ijcai.org/Proceedings/2019/0114.pdf) |
2019|IJCAI 2019| Coarse-to-Fine Image Inpainting via Region-wise Convolutionsand Non-Local Correlation [[pdf]](https://www.ijcai.org/Proceedings/2019/0433.pdf) |
2019|ACM MM 2019| Progressive Image Inpainting with Full-Resolution Residual Network [[pdf]](https://arxiv.org/pdf/1907.10478.pdf) [[code]](https://github.com/ZongyuGuo/Inpainting_FRRN)|
2019|ACM MM 2019| Deep Fusion Network for Image Completion [[pdf]](https://arxiv.org/pdf/1904.08060.pdf) [[code]](https://github.com/hughplay/DFNet)| CNN-based Fusion Block
2019|ACM MM 2019| GAIN: Gradient Augmented Inpainting Network for Irregular Holes [[pdf]](https://dl.acm.org/citation.cfm?id=3350912) |
2019|ACM MM 2019| Single-shot Semantic Image Inpainting with Densely Connected Generative Networks [[pdf]](https://dl.acm.org/citation.cfm?id=3350903) |
2019|ICCVW 2019|EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning [[pdf]](http://openaccess.thecvf.com/content_ICCVW_2019/papers/AIM/Nazeri_EdgeConnect_Structure_Guided_Image_Inpainting_using_Edge_Prediction_ICCVW_2019_paper.pdf) [[code]](https://github.com/knazeri/edge-connect)|
2019|ICCV 2019|Coherent Semantic Attention for Image Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Coherent_Semantic_Attention_for_Image_Inpainting_ICCV_2019_paper.pdf) [[code]](https://github.com/KumapowerLIU/CSA-inpainting)|
2019|ICCV 2019|StructureFlow: Image Inpainting via Structure-aware Appearance Flow [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ren_StructureFlow_Image_Inpainting_via_Structure-Aware_Appearance_Flow_ICCV_2019_paper.pdf) [[code]](https://github.com/RenYurui/StructureFlow) |
2019|ICCV 2019|Progressive Reconstruction of Visual Structure for Image Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Progressive_Reconstruction_of_Visual_Structure_for_Image_Inpainting_ICCV_2019_paper.pdf) [[code]](https://github.com/jingyuanli001/PRVS-Image-Inpainting) |
2019|ICCV 2019| Localization of Deep Inpainting Using High-Pass Fully Convolutional Network [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Localization_of_Deep_Inpainting_Using_High-Pass_Fully_Convolutional_Network_ICCV_2019_paper.pdf)|
2019|ICCV 2019| Image Inpainting with Learnable Bidirectional Attention Maps [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xie_Image_Inpainting_With_Learnable_Bidirectional_Attention_Maps_ICCV_2019_paper.pdf) [[code]](https://github.com/Vious/LBAM_inpainting) |
2019|ICCV 2019|Free-Form Image Inpainting with Gated Convolution [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Free-Form_Image_Inpainting_With_Gated_Convolution_ICCV_2019_paper.pdf) [[project]](http://jiahuiyu.com/deepfill2/)|
2019|ICCV 2019| CIIDefence: Defeating Adversarial Attacks by Fusing Class-specific Image Inpainting and Image Denoising [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gupta_CIIDefence_Defeating_Adversarial_Attacks_by_Fusing_Class-Specific_Image_Inpainting_and_ICCV_2019_paper.pdf)|
2019|ICCV 2019| FiNet: Compatible and Diverse Fashion Image Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_FiNet_Compatible_and_Diverse_Fashion_Image_Inpainting_ICCV_2019_paper.pdf) | Fashion
2019|ICCV 2019|SC-FEGAN: Face Editing Generative Adversarial Network with User's Sketch and Color [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jo_SC-FEGAN_Face_Editing_Generative_Adversarial_Network_With_Users_Sketch_and_ICCV_2019_paper.pdf) [[code]](https://github.com/JoYoungjoo/SC-FEGAN)  | Face
2019|ICCV 2019| Human Motion Prediction via Spatio-Temporal Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Hernandez_Human_Motion_Prediction_via_Spatio-Temporal_Inpainting_ICCV_2019_paper.pdf) [[code]](https://github.com/magnux/MotionGAN) | Motion
2019|ICCV 2019| Copy-and-Paste Networks for Deep Video Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Copy-and-Paste_Networks_for_Deep_Video_Inpainting_ICCV_2019_paper.pdf) | Video
2019|ICCV 2019 |Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chang_Free-Form_Video_Inpainting_With_3D_Gated_Convolution_and_Temporal_PatchGAN_ICCV_2019_paper.pdf) [[code]](https://github.com/amjltc295/Free-Form-Video-Inpainting) | Video
2019|ICCV 2019| An Internal Learning Approach to Video Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_An_Internal_Learning_Approach_to_Video_Inpainting_ICCV_2019_paper.pdf) | Video
2019|ICCV 2019| Vision-Infused Deep Audio Inpainting [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Vision-Infused_Deep_Audio_Inpainting_ICCV_2019_paper.pdf) [[code]](https://github.com/Hangz-nju-cuhk/Vision-Infused-Audio-Inpainter-VIAI) | Audio
2019|ACCESS 2019|Artist-Net: Decorating the Inferred Content With Unified Style for Image Inpainting [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8669751)|CNN-based
2019|NPL 2019|An Improved Method for Semantic Image Inpainting with GANs: Progressive Inpainting [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs11063-018-9877-6.pdf)|GAN-based
2019|NPL 2019|Image inpainting: A review [[paper]](https://arxiv.org/pdf/1909.06399.pdf)|Deep learning
2019|PCS 2019|New Inpainting Algorithm Based on Simplified Context Encoders and Multi-Scale Adversarial Network [[paper]](https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050919X00034/1-s2.0-S1877050919302728/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIGBIgxxHX%2BuyvzUnYZ8L6lkV5jf%2FLOht6tX%2FQoi3S%2BqwAiBelP1a4VXZ9j2lOjPIBwS4XifSc4WaxGgYR6gc1%2FL0PSrPAgguEAIaDDA1OTAwMzU0Njg2NSIMx8neLWTUgTZiqg%2F9KqwC3CvzKQTMn%2FFDGLHBUf89f%2FiFOK%2B1yssQ6UeAJrkAWQCdODfXM3cwFBbTw%2FNOfW8Hdb9s59ObubQ3%2BXx6qkZ369JM%2BcNyaDo8twrCIfWUKcsqKftTu31rIcnCM8HFy1t32ImU%2BFhTF9PoVLnD107HZI8oKGov5diEflQ93kVa2wceYA%2FqSp7ZveLoQcQKDUxbsIqKbOyP8A65JnAa53IE1sKccn8ztYVFy%2FEXy%2FO9cTvgu%2FoH498QG4ws5vJV%2FVjys2RhvxmXfVZnfE8HcrXJ0kqRRJQWsMWqLPR2fQcwQjJiLXuT4fkAGDmeUwg9txsfjj5bipOG00Ls%2F7xiLtacrA2xjQyhYX9xl3%2FbLWPqZaj7W0P2Ph2N5ineNDGynu4C%2F2vT394d3urcGnx9MPetme8FOtECTBRnuUEfm8lj9ruOca%2F%2BNv8K8ofVB%2BWvCnRq9gCHztofhVMJcpQ5qpqWfTClcy8X%2B1JFaBrmb1pi7gYH0eVmGycHMppP5G%2B4S8kj%2BoalsmekjZ1Ig%2FcigYXS58DXmJRbO4FGwaimZU3DEsEaWy4BPLY0foLqX5BWbwJVxn4FzFdes3OGT4c0sXbRlLuqOeOGrtosdGIyusD96G0YbdLnf%2F3ATko7sOwoOlTaNboVYSm2C%2F4HVjetk2mElgMFq0rKaEIrVitiCInOBBV%2FVj52kycZetUR7go0oK4%2Bsoc0pXUB7GPQL%2BJdK%2F%2BjbJRoQLrR7Qjk2QS7M6EKkJ0EhKOm6smhnY3My5hWv0QSPYA5CyWXxkQk2Xvz27X3WrqLc5HU22JFxhoqlWsae86C0hopBZ1m16IIksOymW0RYTAdIzbik7wlugkNtmNntSvdSbAZRQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191203T132958Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6CLGRVOS%2F20191203%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0e37c1929b556d9445f0e0c23920a9e2df5e1becdb9b60a3a1a20da6998afa9b&hash=d4ae3ea53690e2aa23feec59ac3d25f07f8c48e272ac1d815f8a82be89603084&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050919302728&tid=spdf-a6e89a5d-fd2e-4598-9955-8e534f42bbce&sid=651685ba64ca054065189193f652e1a15958gxrqb&type=client)|GAN-based
2019|PRL 2019|Peeking behind objects: Layered depth prediction from a single image [[paper]](https://pdf.sciencedirectassets.com/271524/1-s2.0-S0167865519X00070/1-s2.0-S0167865518307062/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIEas0rIXbRvpR%2F8YIexwuPgWeFlHGpRr3D2MjLRvhUVxAiEA4drtrbw%2F2uecLwYZPoUFkvSWxleGyfBPCb6Eozy4mCUq0AIILhACGgwwNTkwMDM1NDY4NjUiDC6hG7h8xG8OlQY5GyqtAuEya3nF8UprJvazulD3P2PQiMEDGDjtf7BWHluWZhdIu5IXXXoa%2FDv639hzI6WW8BZ3KzDJFIezhBGjUrMpJbjjh4nUxMGrceyk0fr4QAzK3XYc5i3DmnDr7bV%2FASge5XtXbXWbDUxLBjUMzX0rdacAQy2KN9RR155%2FNszUkhAksuJR%2B%2F8CYTCzDPLJQmmJKC6Yj3bP3CLuDNagPnfbi13EZKh%2B0c0ASKBCDhJ8CcoNOrcmD4CVI%2B3JMVrgEMu8MXuj00603f3WScV52qyhU2Z4cF4btxxbJlH1iaR4Hp1SkXNYvzrl7Y8LlR%2BzOmBMAbCCJ9fZyKuF%2BI3M%2F8X%2B7Z4%2F3LB9Ga0l8%2Fw4%2BHIXBDvGuw9k3cJI66%2F4Vrs0C51lSfBymHM%2FephUCocRtw8wva%2BZ7wU6zwI0EuuKkCqrNuAHu2TcjeLz3nT31%2FIdW1GmvipgmkD%2B2%2Buzc87nZG1qmSHFMcqXkFt%2FRwIBvEyR%2FjTJFhPIcwhM1SWbyiNNbwN2PQl926On%2BYljMmuhDvY01CNO3ZcCUTpef1GpWKeaJmAXpxS0Fo%2FyOVIoTKzmgnaOPcPVmgiwXFlni0w0vXmXvSwjHdC7wx%2Fe0JX57Df3d%2Boi5tv8akD2H2QJH1dfWWVKkv26wtgPgONqYa2xLGocJO79HtuxGog6PtnTVTu4aMpIG41fMwVZ3geuody%2BOqguedlXEB5hfXjkxsryvTiKArqwEqcTsvVm5ZYMIsbCQ2db7Bsa7Ya7soN2SEmL0japYzV07B%2ByxlCRIO65yx9AZwZySnOr7lf%2F41UiDuCmCawsHby5zbRwmXxtxe7moriC%2BrNk3kfeBEoDUc0EzygK%2Blx0kPgLiQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191203T134157Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3MBEQ2OK%2F20191203%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c3d4baf4280720e1821f048d71d7681d65f6e8c6bfd9d263af1f021602c235da&hash=23d56d32d76e6194f5b8f66d2762a6d2399a89ae91d14a4a5936be7d1cc1e23b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167865518307062&tid=spdf-30a9ce34-0fa1-4332-972c-4fc28780ea76&sid=bf32171d5938c4482c9ba4c482e4d6e21e87gxrqa&type=client)|GAN-based
2019|VISAPP 2019|Semantic image inpainting through improved Wasserstein generative adversarial networks [[paper]](https://arxiv.org/pdf/1812.01071.pdf)|GAN-based
2020|AAAI 2020|Region Normalization for Image Inpainting [[pdf]](https://arxiv.org/abs/1911.10375) [[code]](https://github.com/geekyutao/RN)|
2020|WACV 2020|Evaluation of Image Inpainting for Classification and Retrieval [[paper]](http://openaccess.thecvf.com/content_WACV_2020/papers/Black_Evaluation_of_Image_Inpainting_for_Classification_and_Retrieval_WACV_2020_paper.pdf)|

## Inpainting Traditional Methods
Year|Proceeding|Title|Comment
--|:--:|:--:|:--
2000|SIGGRAPH 2000|Image Inpainting  [[pdf]](http://slipguru.disi.unige.it/readinggroup/papers_vis/bertalmio00inpainting.pdf) [[paper]](https://apps.dtic.mil/dtic/tr/fulltext/u2/a437378.pdf) |Diffusion-based
2001|TIP 2001|Filling-in by joint interpolation of vector fields and gray levels [[pdf]](https://conservancy.umn.edu/bitstream/handle/11299/3462/1/1706.pdf)|Diffusion-based
2001|CVPR 2001|Navier-stokes, ﬂuid dynamics, and image and video inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=990497)|
2001|SIGGRAPH 2001|Image Quilting for Texture Synthesis and Transfer [[pdf]](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf)  |
2001|SIGGRAPH 2001|Synthesizing Natural Textures [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.359.8241&rep=rep1&type=pdf)|
2002|EJAM 2002|Digital inpainting based on the mumford–shah–euler image model [[pdf]](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/26ACC4694C7F064B6F40D55C09ACA9A1/S0956792502004904a.pdf/digital_inpainting_based_on_the_mumfordshaheuler_image_model.pdf)  |Diffusion-based
2003|CVPR 2003| Object removal by exemplar-based inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1211538)|
2003|TIP 2003|Simultaneous structure and texture image inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1217265)|Diffusion-based
2003|TIP 2003|Structure and Texture Filling-In of Missing Image Blocks in Wireless Transmission and Compression Applications [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1197835)|
2003|ICCV 2003|Learning How to Inpaint from Global Image Statistics [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1238360)|Diffusion-based
2003|TOG 2003|Fragment-based image completion [[pdf]](http://delivery.acm.org/10.1145/890000/882267/p303-drori.pdf?ip=222.195.92.10&id=882267&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EA4F9C023AC60E700%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1553430113_8d3cc7f5adde2fb3894043de791d9150) | Patch-based
2004|TIP 2004|Region Filling and Object Removal by Exemplar-Based Image Inpainting [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_tip2004.pdf)|Patch-based; Inpainting order
2004|TPAMI 2004|Space-Time Video Completion [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1315022) |
2005|SIGGRAPH 2005|Image Completion with Structure Propagation [[pdf]](http://jiansun.org/papers/ImageCompletion_SIGGRAPH05.pdf)|Patch-based
2006|ISCS 2006|Image Compression with Structure Aware Inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1692960)|
2006|JIVP 2006|Spatio-temporal consistent depth-image-based rendering using layered depth image and inpainting [[paper]](https://link.springer.com/content/pdf/10.1186/s13640-016-0109-6.pdf)| Object removal
2007|TOG 2007| Scene completion using millions of photographs [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.299.518&rep=rep1&type=pdf)|
2007|CSVT 2007|Image Compression With Edge-Based Inpainting [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/inpainting_csvt_07.pdf)|Diffusion-based
2007|TPAMI 2007|Space-Time Video Completion [[pdf]](http://www.wisdom.weizmann.ac.il/~vision/VideoCompletion/SpaceTimeCompletion.pdf)| video
2008|CVPR 2008|Summarizing Visual Data Using Bidirectional Similarity [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4587842)|
2009|SIGGRAPH 2009|PatchMatch: a randomized correspondence algorithm for structural image editing [[pdf]](http://www.faculty.idc.ac.il/arik/seminar2009/papers/patchMatch.pdf)  |Patch-based
2010| TIP 2010|Image inpainting by patch propagation using patch sparsity [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5404308) |Patch-based
2011|FTCGV 2011|Structured learning and prediction in computer vision [[pdf]](http://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf)|
2011|ICIP 2011|Examplar-based inpainting based on local geometry [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6116441)|Inpainting order
2012|TOG 2012|Combining inconsistent images using patch-based synthesis [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.364.5147&rep=rep1&type=pdf)|Patch-based
2012|ACM TOG 2012|Image Melding: Combining Inconsistent Images using Patch-based Synthesis [[pdf]](https://www.ece.ucsb.edu/~psen/Papers/SIGGRAPH12_ImageMelding.pdf)|Patch-based
2014|TOG 2014|Image completion using Planar structure guidance [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/structure_completion_small.pdf)|Patch-based
2014|TIP 2014|Context-aware patch-based image inpainting using markov random field modeling [[paper]](https://telin.ugent.be/~sanja/Papers/Inpainting/Inpainting_TIP2014_RuzicPizurica.pdf) [[code]](https://github.com/nimpy/inpynting)|Patch-based
2014|TVCG 2014|High-Quality Real-Time Video Inpainting with PixMix [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6714519)|Video
2014|SIAM 2014|Video inpainting of complex scenes [[pdf]](https://arxiv.org/pdf/1503.05528.pdf)|Video
2015|TIP 2015|Annihilating Filter-Based Low-Rank Hankel Matrix Approach for Image Inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7127011)|
2015|TIP 2015|Exemplar-Based Inpainting: Technical Review and New Heuristics for Better Geometric Reconstructions [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7056453)|
2015|TVCG 2015|Diminished reality based on image inpainting considering background geometry [[paper]](library.naist.jp/dspace/bitstream/handle/10061/11030/1427_TVCG_DR_author_ver.pdf?sequence=1&isAllowed=y)|
2015|JGO 2015|Fast algorithm for color texture image inpainting using the non-local ctv model [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10898-015-0290-7.pdf)|Patch-based
2016|MTA 2016|Rate-distortion optimized image compression based on image inpainting [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs11042-014-2332-4.pdf)|Patch-based
2016|SIVP 2016|Domain-based structure-aware image inpainting [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs11760-015-0840-y.pdf)|Patch-based
2016|SC 2016|Image inpainting algorithm based on TV model and evolutionary algorithm [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs00500-014-1547-7.pdf)|Diffusion-based
2016|ICAC 2016|Image Inpainting Approaches – A Review [[paper]](https://www.researchgate.net/profile/Smriti_Bhandari2/publication/306310171_Image_Inpainting_Approaches_-_A_Review/links/5bd98d37a6fdcc3a8db3066d/Image-Inpainting-Approaches-A-Review.pdf)|
2016|TOG 2016|Temporally coherent completion of dynamic video [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/SigAsia_2016_VideoCompletion.pdf) | Video
2017|TVCG 2017|Patch-Based Image Inpainting via Two-Stage Low Rank Approximation [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7922581&tag=1)|Patch-based
2017|TIP 2017|Depth image inpainting: Improving low rank matrix completion with low gradient regularization [[paper]](https://arxiv.org/pdf/1604.05817.pdf) [[code]](https://github.com/ZJULearning/depthInpainting)|Patch-based
2017|MTA 2017|Exemplar-based image inpainting using svd-based approximation matrix and multi-scale analysis [[paper]](http://journal.iis.sinica.edu.tw/paper/1/100484-2.pdf?cd=7BA29DFFD1BE9BF88)|Patch-based
2017|EURASIP 2017|Damaged region filling and evaluation by symmetrical exemplar-based image inpainting for Thangka [[paper]](https://jivp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13640-017-0186-1?site=jivp-eurasipjournals.springeropen.com)| Patch-based
2017|TIFS 2017|Localization of diffusion-based inpainting in digital images [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7987733)|Diffusion-based
2018|MTA 2018|Gradient-based low rank method and its application in image inpainting [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs11042-017-4509-0.pdf)|Patch-based
2018|MTA 2018|A novel patch matching algorithm for exemplar-based image inpainting [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs11042-017-5077-z.pdf)|Patch-based
2018|MTA 2018|A robust forgery detection algorithm for object removal by exemplar-based image inpainting [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs11042-017-4829-0.pdf)|Patch-based
2018|TIP 2018|Image Inpainting Using Nonlocal Texture Matching and Nonlinear Filtering [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8531678)|Patch-based
2018|TMM 2018|Structure-guided image inpainting using homography transformation [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8352516&tag=1)|Patch-based
2018|CC 2018|Damaged region filling by improved  criminisi image inpainting algorithm for thangka [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10586-018-2068-4.pdf)|Patch-based
2018|CC 2018|The research of image inpainting algorithm using self-adaptive group structure and sparse representation [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10586-018-2323-8.pdf)|dictionary learning-based
2018|ACCESS 2018|Sparsity-based image inpainting detection via canonical correlation analysis with low-rank constraints [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8439932)|Sparsity-based
2019|AJSE 2019|Image Inpainting Algorithm Based on Saliency Map and Gray Entropy [[paper]](https://jivp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13640-017-0186-1)|Patch-based
2019|CSSP 2019|Image inpainting based on fractional-order nonlinear diffusion for image reconstruction [[paper]](https://link.springer.com/content/pdf/10.1007%2Fs00034-019-01029-w.pdf)|Diffusion-based

## ArXiv or Unpublished
Year|Proceeding|Title|Comment
--|:--:|:--:|:--
2018|arXiv:1801.07632|High Resolution Face Completion with Multiple Controllable Attributes via Fully End-to-End Progressive Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1801.07632.pdf)|
2018|arXiv:1803.07422|Patch-Based Image Inpainting with Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1803.07422.pdf) [[code]](https://github.com/nashory/pggan-pytorch)|
2018|arXiv:1808.04432|X-GANs: Image Reconstruction Made Easy for Extreme Cases [[pdf]](https://arxiv.org/pdf/1808.04432.pdf)|
2018|arXiv:1811.03911| Unsupervised Learnable Sinogram Inpainting Network (SIN) for Limited Angle CT reconstruction [[paper]](http://export.arxiv.org/pdf/1811.03911)|CNN-based
2018|arXiv:1811.07104|On Hallucinating Context and Background Pixels from a Face Mask using Multi-scale GANs [[pdf]](https://arxiv.org/pdf/1811.07104.pdf)|
2018|arXiv:1811.09012|Multi-View Inpainting for RGB-D Sequence [[pdf]](https://arxiv.org/pdf/1811.09012.pdf)|
2018|arXiv:1812.01458|Deep Inception Generative network for Cognitive Image Inpainting [[pdf]](https://arxiv.org/pdf/1812.01458.pdf)|
2018|arXiv:1808.08483|Painting Outside the Box: Image Outpainting with GANs [[paper]](https://arxiv.org/pdf/1808.08483.pdf)|GAN-based
2019|arXiv:1901.03396|Detecting Overfitting of Deep Generative Networks via Latent Recovery [[pdf]](https://arxiv.org/pdf/1901.03396.pdf)|
2019|arXiv:1902.00301|Deep Hyperspectral Prior: Denoising, Inpainting, Super-Resolution [[paper]](https://arxiv.org/pdf/1902.00301)|CNN-based
2019|arXiv:1902.09225|Harmonizing Maximum Likelihood with GANs for Multimodal Conditional Generation [[pdf]](https://arxiv.org/pdf/1902.09225.pdf)|
2019|arXiv:1903.00450|Multi-Object Representation Learning with Iterative Variational Inference [[pdf]](https://arxiv.org/pdf/1903.00450.pdf) |
2019|arXiv:1903.04842 |Unsupervised motion saliency map estimation based on optical flow inpainting [[pdf]](https://arxiv.org/pdf/1903.04842.pdf)|
2019|arXiv:1903.10885|Learning Quadrangulated Patches For 3D Shape Processing [[pdf]](https://arxiv.org/pdf/1903.10885.pdf) |
2019|arXiv:1904.10795 |Graph-based Inpainting for 3D Dynamic Point Clouds [[pdf]](https://arxiv.org/pdf/1904.10795.pdf)| Point Cloud
2019|arXiv:1905.02882 |Frame-Recurrent Video Inpainting by Robust Optical Flow Inference [[pdf]](https://arxiv.org/pdf/1905.02882.pdf)| Video
2019|arXiv:1905.02949 |Deep Blind Video Decaptioning by Temporal Aggregation and Recurrence [[pdf]](https://arxiv.org/pdf/1905.02949.pdf)| Video
2019|arXiv:1905.12384|Coherent Semantic Attention for Image Inpainting [[paper]](https://arxiv.org/pdf/1905.12384.pdf)|CNN-based
2019|arXiv:1905.13066|Align-and-Attend Network for Globally and Locally Coherent Video Inpainting [[pdf]](https://arxiv.org/pdf/1905.13066.pdf)| Video
2019|arXiv:1906.00884|Fashion Editing with Multi-scale Attention Normalization [[pdf]](https://arxiv.org/pdf/1906.00884.pdf)|
2019|arXiv:1912.01834|Diversity-Generated Image Inpainting with Style Extraction [[paper]](https://arxiv.org/pdf/1912.01834.pdf) [[code]](https://github.com/vivitsai/SEGAN#)|GAN-based
2019|arXiv:1912.11711|Controllable and Progressive Image Extrapolation [[paper]](https://arxiv.org/pdf/1912.11711.pdf)|GAN-based
2020|arXiv:1912.10960v2|Image Outpainting and Harmonization using Generative Adversarial Networks [[paper]](https://arxiv.org/pdf/1912.10960v2.pdf) [[code]](https://github.com/basilevh/image-outpainting)|GAN-based
2020|arXiv:2001.04568| 180-degree Outpainting from a Single Image [[paper]](https://arxiv.org/pdf/2001.04568.pdf)|GAN-based


# Related Link
[ICCV 2019](http://openaccess.thecvf.com/ICCV2019.py)

[CVPR 2019](http://openaccess.thecvf.com/CVPR2019.py)

[ECCV 2018](http://openaccess.thecvf.com/ECCV2018.py)

[ECCV 90-2018](https://link.springer.com/conference/eccv)

[NeurlPS 1988-2019](http://papers.nips.cc/)

[ICLR 2019](https://iclr.cc/Conferences/2019/Schedule?type=Poster)

[ICML 2019](https://icml.cc/Conferences/2019/Schedule?type=Poster)

[AAAI 2019]( https://aaai.org/Conferences/AAAI-19/wp-content/uploads/2018/11/AAAI-19_Accepted_Papers.pdf)

[IJCAI 2019](https://www.ijcai19.org/accepted-papers.html)

[ACL 2019](http://www.acl2019.org/EN/program.xhtml)
