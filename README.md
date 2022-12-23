# Rec-Models

📝 Summary of recommendation, advertising and search models.

## Recall



## Ranking


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



## Reranking



## Learning Resource

- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- [EasyRec](https://github.com/alibaba/easyrec)
- [AI-RecommenderSystem](https://github.com/zhongqiangwu960812/AI-RecommenderSystem)
- [DeepMatch](https://github.com/shenweichen/DeepMatch)









