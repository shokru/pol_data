Supervised Learning with political data
================
Guillaume Coqueret

This repository contains teaching and research material focused on supervised learning with political data.

The original database is featured in [RData](https://github.com/shokru/pol_data/blob/master/anes.RData) format and in [CSV](https://github.com/shokru/pol_data/blob/master/anes.csv) format. The RData file is more complete because (ordered) categories are coded as (ordered) factors. The code is written in R and tutorial is split in three sets of scripts:

1.  **[Datavisualization](https://github.com/shokru/pol_data/blob/master/VIZ.md)**: plotting distribution, trends, etc.

2.  **[Classification exercises](https://github.com/shokru/pol_data/blob/master/CLASS.md)**: try to predict the party affiliation based on 6 variables.
    *Methods used*: simple trees, boosted trees, support vector machines, neural networks (multilayer perceptron).

3.  **[Regression analysis](https://github.com/shokru/pol_data/blob/master/REG.md)**: try to predict the feeling towards unions using 9 variables.
    *Methods used*: linear regression, simple trees, boosted trees, support vector machines, neural networks (multilayer perceptron).

For each task and each technique, the degrees of freedom are numerous and the purpose is to play with the choice of variables, the parameters and the hyper-parameters.

Whenever using this material, please cite the original collector of the data, ANES, and the related paper *[Supervised learning with political data](https://github.com/shokru/pol_data/blob/master/Supervised%20learning%20with%20political%20data.pdf)* by Guillaume Coqueret.

Any questions / comments / requests can be sent to guillaume.coqueret"at"gmail.com.
