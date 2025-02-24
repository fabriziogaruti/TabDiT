# Diffusion Transformers for Tabular Data Time Series Generation
Official PyTorch implementaton of paper Diffusion Transformers for Tabular Data Time Series Generation.  

## Introduction
Tabular data generation has recently attracted a growing interest due to its different application scenarios. However, generating *time series* of tabular data, where each element of the series depends on the others, remains a largely unexplored domain. 
This gap is probably due to the difficulty of jointly solving different problems, the main of which are the heterogeneity of tabular data (a problem common to non-time-dependent approaches) and the variable length of a time series.
In this paper, we propose a Diffusion Transformers (DiTs) based approach for tabular data series generation. Inspired by the recent success of DiTs in image and video generation, we extend this framework to deal with heterogeneous data and variable-length sequences. 
Using extensive experiments on six datasets, we show that the proposed approach outperforms previous work by a large margin.
