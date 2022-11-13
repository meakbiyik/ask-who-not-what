# Ask "Who", Not "What": Bitcoin Volatility Forecasting with Twitter Data

This repository contains the code, document and references for the paper "Ask "Who", Not "What": Bitcoin Volatility Forecasting with Twitter Data" by M. Eren Akbiyik, Mert Erkul, Killian Kämpf, Dr. Vaiva Vasiliauskaite and Dr. Nino Antulov-Fantulin at ETH Zürich. This work is published in the Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (WSDM '23).

## Dataset

The Tweet dataset parsed and used as part of this work is publicly available [here](https://sobigdata.d4science.org/catalogue-sobigdata?path=/dataset/crypto_related_tweets_from_10_10_2020_to_3_3_2021) under the license CC BY 4.0. The dataset contains 30 million cryptocurrency-related tweets from 10.10.2020 to 3.3.2021.

## Citation

If you use this code or the dataset, please cite our paper.

### ACM Reference Format

```acm
M. Eren Akbiyik, Mert Erkul, Killian Kämpf, Vaiva Vasiliauskaite, and Nino Antulov-Fantulin. 2023. Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data. In *Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (WSDM ’23), February 27–March 3, 2023, Singapore, Singapore*. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3539597.3570387
```

### BibTeX

```bibtex
@inproceedings{10.1145/3539597.3570387,
    author = {Akbiyik, M. Eren and Erkul, Mert and K{\"a}mpf, Killian and Vasiliauskaite, Vaiva and Antulov-Fantulin, Nino},
    title = {Ask “Who”, Not “What”: Bitcoin Volatility Forecasting with Twitter Data},
    year = {2023},
    isbn = {978-1-4503-9407-9/23/02},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3539597.3570387},
    doi = {10.1145/3539597.3570387},
    booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
    pages = {9},
    numpages = {9},
    keywords = {Bitcoin, Twitter, Volatility Forecasting},
    location = {Singapore, Singapore},
    series = {WSDM ’23}
}
```

## Abstract

Understanding the variations in trading price (volatility), and its response to exogenous information, is a well-researched topic in finance. In this study, we focus on finding stable and accurate volatility predictors for a relatively new asset class of cryptocurrencies, in particular Bitcoin, using deep learning representations of public social media data obtained from Twitter. For our experiments, we extracted semantic information and user statistics from over 30 million Bitcoin-related tweets, in conjunction with 15-minute frequency price data over a horizon of 144 days. Using this data, we built several deep learning architectures that utilized different combinations of the gathered information. For each model, we conducted ablation studies to assess the influence of different components and feature sets over the prediction accuracy. We found statistical evidences for the hypotheses that: (i) temporal convolutional networks perform significantly better than both classical autoregressive models and other deep learning-based architectures in the literature, and (ii) tweet author meta-information, even detached from the tweet itself, is a better predictor of volatility than the semantic content and tweet volume statistics. We demonstrate how different information sets gathered from social media can be utilized in different architectures and how they affect the prediction results. As an additional contribution, we make our dataset public for future research.

## Acknowledgements

M.E.A., M.E. and K.K. thank Prof.\ Dr.\ Ce Zhang for their help during this research, Benjamin Suter for his help on the collected tweet dataset, and the the ETH Z\"urich DS3Lab for giving us access to their computer infrastructure. N.A.F. and V.V. are supported by the European Union - Horizon 2020 Program under the scheme ‘INFRAIA-01-2018-2019 - Integrating Activities for Advanced Communities’, Grant Agreement no. 871042, ‘SoBigData++: European Integrated Infrastructure for Social Mining and Big Data Analytics’ (<http://www.sobigdata.eu>).