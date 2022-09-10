## Complex affect dynamics
This project aims to shed light on how the mechanism of emotions is induced in an external observer of a dyadic conversation using the emotion annotations of [KEmoCon](https://www.nature.com/articles/s41597-020-00630-y) paper and via the analysis of complex affect dynamics.

The folder *MATLAB code* contains the code that was developed for the following paper: 
- **Dejonckheere, E., Mestdagh, M., Houben, M. et al. Complex affect dynamics add limited information to the prediction of psychological well-being. Nat Hum Behav 3, 478–491 (2019), https://doi.org/10.1038/s41562-019-0555-0.**

### Positive & Negative Affect time series calculation 
For the calculation of Positive Affect (PA) and Negative Affect (NA) time series for each participant we used the following emotion
annotation categories:
- Cheerful, Happy for PA
- Angry, Nervous, Sad for NA 

where the measurement scale is from 1: very low to 4: very high, respectively, as described in detail in KEmoCon paper. The unique value of PA and NA for each annotation step is derived as the mean of the individual emotional states belonging to each category. This value is then transformed from the present range ([1, 4]) to a larger ([1, 100]). The result is the creation of a time series for each emotional state, experimenter, and annotation perspective.

### Complex affect dynamic measures
For the calculation of complex affect dynamic measures, we utilized the paper mentioned above. After preprocessing the KEmoCon dataset to the appropriate format instructed by the authors of the paper above via [preprocess_dataset.py](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/preprocess_dataset.py), we
calculated the following complex affect dynamic measures for each participant and for
each annotation perspective: 
- Mean PA and NA (**M**), which captures one’s average level of positive or negative affect.
- Variance or std. PA and NA (**s.d**), which captures the average emotional deviation from one’s mean levels of positive or negative affect.
- Relative variance or standard deviation PA and NA (**s.d***), which captures the average emotional deviation from one’s mean levels of positive or negative affect, taking into account the maximum possible variability given the mean of that affective state.
- MSSD PA and NA (**MSSD**), which captures the average change in emotional intensity between two successive measurement occasions for positive or negative affect.
- Auto-regressive slope of PA and NA (**AR**), which captures the degree to which positive or negative affect carries over from one moment to the next, is self-predictive, and resistant to change.

The plots below illustrate the comparison between the distributions of each dynamic measure among the 32 participants for each annotation perspective.
Observing them, we come to the conclusion that all external observers perceive the average level of positive and negative emotional states differently than the participants engaged in the conversation. At the same time, a difference is observed in the distributions of the autoregressive slope (AR) between the external observers and participants. Therefore, either the external observers fail to fully grasp the sequence of emotions as they do not participate in the discussion, either a subset of the observers are biased towards the intensity of certain individual emotional states.

Mean PA - self vs partner         |  Mean PA - self vs external         |  Mean PA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_mean_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_mean_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_mean_partner_external.png)


Mean NA - self vs partner         |  Mean NA - self vs external         |  Mean NA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_mean_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_mean_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_mean_partner_external.png)

s.d. PA - self vs partner         |  s.d. PA - self vs external         |  s.d. PA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_variance_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_variance_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_variance_partner_external.png)

s.d. NA - self vs partner         |  s.d. NA - self vs external         |  s.d. NA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_variance_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_variance_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_variance_partner_external.png)


relative s.d. PA - self vs partner         | relative s.d. PA - self vs external         | relative s.d. PA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_relative_variance_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_relative_variance_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_relative_variance_partner_external.png)


relative s.d. NA - self vs partner         | relative s.d. NA - self vs external         | relative s.d. NA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_relative_variance_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_relative_variance_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_relative_variance_partner_external.png)


MSSD PA - self vs partner         | MSSD PA - self vs external         | MSSD PA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_MSSD_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_MSSD_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_MSSD_partner_external.png)


MSSD NA - self vs partner         | MSSD NA - self vs external         | MSSD NA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_MSSD_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_MSSD_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_MSSD_partner_external.png)


AR PA - self vs partner         | AR PA - self vs external         | AR PA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_AR_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_AR_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/PA_AR_partner_external.png)


AR NA - self vs partner         | AR NA - self vs external         | AR NA - partner vs external
:-------------------------:|:-------------------------: | :-------------------------:
![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_AR_self_partner.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_AR_self_external.png)  |  ![](https://github.com/sotirismos/Complex-Affect-Dynamics/blob/main/results/NA_AR_partner_external.png)
