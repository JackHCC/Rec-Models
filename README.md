# Rec-Models

📝 Summary of recommendation, advertising and search models.

## Recall

### Papers

| Paper                                                        | Resource | Others                                                       |
| :----------------------------------------------------------- | -------- | ------------------------------------------------------------ |
| [2019阿里SDM模型] *[SDM: Sequential Deep Matching Model for Online Large-scale Recommender System](https://arxiv.org/pdf/1905.03028v2.pdf)* |          | [Code](https://github.com/alicogintel/SDM)                   |
| [2019阿里JTM] *[Joint Optimization of Tree-based Index and Deep Model for Recommender Systems](https://arxiv.org/pdf/1902.07565v1.pdf)* |          | [Code](https://github.com/massquantity/dismember)            |
| [2019百度MOBIUS] [*MOBIUS:Towards the Next Generation of Qery Ad Matching in Baidu's Sponsored Search*](http://research.baidu.com/Public/uploads/5d12eca098d40.pdf) |          | Code                                                         |
| [2019YouTube双塔]  *[sampling bias corrected neural modeling for large corpus item recommendations](https://dl.acm.org/doi/abs/10.1145/3298689.3346996)* |          | Code                                                         |
| [2018阿里TDM] [*Learning Tree-based Deep Model for Recommender Systems*](https://arxiv.org/pdf/1801.02294.pdf) |          | [Code](https://github.com/alibaba/x-deeplearning)            |
| [2018Facebook] [*Collaborative Multi-modal deep learning for the personalized product retrieval in Facebook Marketplace*](https://arxiv.org/pdf/1805.12312.pdf) |          | Code                                                         |
| [2013 DSSM模型] *[Learning deep structured semantic models for web search using clickthrough data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)* |          | [Code](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.1.0/models/match/dssm) |
| [2008 SVD] [*Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model*](https://people.engr.tamu.edu/huangrh/Spring16/papers_course/matrix_factorization.pdf) |          | Code                                                         |
| [2008] *[Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)* |          | Code                                                         |

## Ranking（CTR|CVR）

### Papers


|                 Model                  | Paper                                                        | Resource                                                     | Others                                                       |
| :------------------------------------: | :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf) | [*CCPM*-基于卷积的点击预测模型](https://zhuanlan.zhihu.com/p/159012746) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/ccpm.py) |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf) |                                                              | Code                                                         |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf) | [*PNN*论文笔记](https://zhuanlan.zhihu.com/p/105084140)      | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/pnn.py) |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) | [*Wide&Deep*模型](https://zhuanlan.zhihu.com/p/94614455)     | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/wdl.py) |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf) | [深度推荐模型之*DeepFM*](https://zhuanlan.zhihu.com/p/57873613) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/deepfm.py) |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194) | [MLR算法模型](https://zhuanlan.zhihu.com/p/77798409?utm_source=wechat_session) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/mlr.py) |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | [谷歌经典 Deep&Cross Network原理](https://zhuanlan.zhihu.com/p/368381633) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/dcn.py) |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) | [推荐算法精排模型*AFM：Attentional Factorization* Machines](https://zhuanlan.zhihu.com/p/395140453) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/afm.py) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf) | [*NFM* 模型 (论文精读)--广告&推荐](https://zhuanlan.zhihu.com/p/42392091) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/nfm.py) |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) | [*xDeepFM* 原理通俗解释及代码实战](https://zhuanlan.zhihu.com/p/371849616) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/xdeepfm.py) |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) | [阿里巴巴*DIN*模型详解](https://zhuanlan.zhihu.com/p/103552262) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/din.py) |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf) | [*DIEN*算法学习笔记](https://zhuanlan.zhihu.com/p/463652456) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/dien.py) |
|                AutoInt                 | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | [*AutoInt*：基于Multi-Head Self-Attention构造高阶特征](https://zhuanlan.zhihu.com/p/60185134) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/autoint.py) |
|                  ONN                   | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf) | [*ONN*: paper+code reading](https://zhuanlan.zhihu.com/p/80830028) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/onn.py) |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf) | [*FiBiNET*: paper reading + 实践调优经验](https://zhuanlan.zhihu.com/p/79659557) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/fibinet.py) |
|                  IFM                   | [IJCAI 2019][An Input-aware Factorization Machine for Sparse Prediction](https://www.ijcai.org/Proceedings/2019/0203.pdf) | [*IFM*: 输入感知的FM*模型*](https://zhuanlan.zhihu.com/p/378615059) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/ifm.py) |
|                 DCN V2                 | [arxiv 2020][DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) | [DCNMix原理与实践](https://zhuanlan.zhihu.com/p/352110578)   | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/dcnmix.py) |
|                  DIFM                  | [IJCAI 2020][A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf) | [DIFM: 双重IFM模型](https://zhuanlan.zhihu.com/p/378619211)  | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/difm.py) |
|                  AFN                   | [AAAI 2020][Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://arxiv.org/pdf/1909.03276) |                                                              | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/afn.py) |
|              SharedBottom              | [arxiv 2017][An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf) | [Shared-Bottom网络结构](http://www.hbase.cn/archives/560.html) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/sharedbottom.py) |
|                  ESMM                  | [SIGIR 2018][Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://dl.acm.org/doi/10.1145/3209978.3210104) | [ESMM](https://www.baidu.com/link?url=N-MIK4n55xdIyPx_aHlS-LlzErnpypa9kQ9A1_dFocYFnImARAnVAuiAtDz6kcu9&wd=&eqid=b72b2f8d0003d4340000000363a54dde)[详解](https://www.baidu.com/link?url=N-MIK4n55xdIyPx_aHlS-LlzErnpypa9kQ9A1_dFocYFnImARAnVAuiAtDz6kcu9&wd=&eqid=b72b2f8d0003d4340000000363a54dde) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/esmm.py) |
|                  MMOE                  | [KDD 2018][Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007) | [多任务学习之MMOE模型](https://zhuanlan.zhihu.com/p/145288000) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/mmoe.py) |
|                  PLE                   | [RecSys 2020][Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236) | [腾讯PCG RecSys2020最佳长论文——视频推荐场景下多任务*PLE模型*详解](https://zhuanlan.zhihu.com/p/272708728) | [Code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/ple.py) |

### Datasets

- [Criteo](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset)
- [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data)
- [Movielens](https://grouplens.org/datasets/movielens/)
- [Amazon](http://jmcauley.ucsd.edu/data/amazon/)
- [Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/dataset/408)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)



## Reranking

### Papers

| Paper                                                        | Resource | Others                                                       |
| :----------------------------------------------------------- | -------- | ------------------------------------------------------------ |
| [IJCAJ2018, Alibaba]. *[Globally Optimized Mutual Influence Aware Ranking in E-Commerce Search](https://arxiv.org/pdf/1805.08524.pdf)* |          | Code                                                         |
| [SIGIR2018, Qingyao Ai]. *[Learning a Deep Listwise Context Model for Ranking Refinement](https://dl.acm.org/doi/pdf/10.1145/3209978.3209985)* |          | [Code](https://github.com/QingyaoAi/Deep-Listwise-Context-Model-for-Ranking-Refinement) |
| [RecSys2019, Alibaba]. *[Personalized Re-ranking for Recommendation](https://arxiv.org/pdf/1904.06813.pdf)* |          | Code                                                         |
| [CIKM2020, Alibaba]. *[EdgeRec-Recommender System on Edge in Mobile Taobao](https://arxiv.org/pdf/2005.08416.pdf)* |          | Code                                                         |
| [Artix2021, Alibaba]. *[Revisit Recommender System in the Permutation Prospective](https://arxiv.org/pdf/2102.12057.pdf)* |          | Code                                                         |

### Blogs

- [基于上下文感知的重排序算法梳理](https://fly-adser.top/2022/03/06/rerankalg/)



## Calibration

### Papers

| Paper                                                        | Resource | Others                                                       |
| :----------------------------------------------------------- | -------- | ------------------------------------------------------------ |
| (KDD2020, Alibaba). *[Calibrating User Response Predictions in Online Advertising](https://link.springer.com/chapter/10.1007/978-3-030-67667-4_13)* |          | Code                                                         |
| (WWW2020, Tencent). *[A Simple and Empirically Strong Method for Reliable Probabilistic Predictions](https://arxiv.org/pdf/1905.10713v3.pdf)* |          | Code                                                         |
| (WWW2022, Alibaba). *[MBCT Tree-Based Feature-Aware Binning for Individual Uncertainty Calibration](https://arxiv.org/pdf/2202.04348v1.pdf)* |          | [Code](https://github.com/huangsg1/Tree-Based-Feature-Aware-Binning-for-Individual-Uncertainty-Calibration) |

### Blogs

- [广告pCTR校准机制](https://fly-adser.top/2022/01/20/ctrcali/)



## Bid

### Papers

| Paper                                                        | Resource | Others                                        |
| :----------------------------------------------------------- | -------- | --------------------------------------------- |
| [IJCAI2017, Alibaba]. *[Optimized Cost per Click in Taobao Display Advertising](https://arxiv.org/pdf/1703.02091v4.pdf)* |          | Code                                          |
| [KDD2019, Alibaba]. *[Bid Optimization by Multivariable Control in Display Advertising](https://arxiv.org/pdf/1905.10928.pdf)* |          | Code                                          |
| [AAMAS2020, ByteDance]. *[Optimized Cost per Mille in Feeds Advertising](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p1359.pdf)* |          | Code                                          |
| [KDD2021, Alibaba]. *[A Unified Solution to Constrained Bidding in Online Display Advertising](https://dl.acm.org/doi/abs/10.1145/3447548.3467199)* |          | Code                                          |
| [KDD2014]. *[Optimal Real-Time Bidding for Display Advertising](http://www0.cs.ucl.ac.uk/staff/w.zhang/papers/ortb-kdd.pdf)* |          | Code                                          |
| [KDD2015]. *[Bid Landscape Forecasting in Online Ad Exchange Marketplace](https://www.researchgate.net/profile/Ruofei-Zhang-2/publication/221653522_Bid_landscape_forecasting_in_online_Ad_exchange_marketplace/links/53f10c1f0cf2711e0c432641/Bid-landscape-forecasting-in-online-Ad-exchange-marketplace.pdf)* |          | Code                                          |
| [KDD2015]. *[Predicting Winning Price in Real Time Bidding with Censored Data](http://wnzhang.net/share/rtb-papers/win-price-pred.pdf)* |          | Code                                          |
| [KDD2016]. [*User Response Learning for Directly Optimizing Campaign Performance in Display Advertising*](https://discovery.ucl.ac.uk/id/eprint/1524035/1/wang_p679-ren.pdf) |          | Code                                          |
| [KDD2016]. [*Functional Bid Landscape Forecasting for Display Advertising*](https://link.springer.com/chapter/10.1007/978-3-319-46128-1_8) |          | [Code](https://github.com/zeromike/bid-lands) |
| [KDD2017]. [*A Gamma-Based Regression for Winning Price Estimation in Real-Time Bidding Advertising*](https://scholar.nycu.edu.tw/zh/publications/a-gamma-based-regression-for-winning-price-estimation-in-real-tim-2) |          | Code                                          |
| [KDD2018]. [*Bidding Machine Learning to Bid for Directly Optimizing Profits in Display Advertising*](https://arxiv.org/pdf/1803.02194.pdf) |          | Code                                          |
| [KDD2019]. *[Deep Landscape Forecasting for Real-time Bidding Advertising](https://arxiv.org/pdf/1905.03028v2.pdf)* |          | [Code](https://github.com/rk2900/DLF)         |

### Blogs

- [RTB论文梳理及总结](https://fly-adser.top/2021/12/29/RTBpapers/)



## Open Resource

- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- [EasyRec](https://github.com/alibaba/easyrec)
- [AI-RecommenderSystem](https://github.com/zhongqiangwu960812/AI-RecommenderSystem)
- [DeepMatch](https://github.com/shenweichen/DeepMatch)
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)
- [RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation)
- [FuxiCTR](https://github.com/xue-pai/FuxiCTR)
- [SELFRec](https://github.com/Coder-Yu/SELFRec)



## Practice in Industry

- [GitHub - Doragd/Algorithm-Practice-in-Industry: 搜索、推荐、广告、用增等工业界实践文章收集（来源：知乎、Datafuntalk、技术公众号）](https://github.com/Doragd/Algorithm-Practice-in-Industry)



