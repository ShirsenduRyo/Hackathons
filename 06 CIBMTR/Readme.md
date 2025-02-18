# Description
Improving survival predictions for allogeneic HCT patients is a vital healthcare challenge. Current predictive models often fall short in addressing disparities related to socioeconomic status, race, and geography. Addressing these gaps is crucial for enhancing patient care, optimizing resource utilization, and rebuilding trust in the healthcare system.

This competition aims to encourage participants to advance predictive modeling by ensuring that survival predictions are both precise and fair for patients across diverse groups. By using synthetic data—which mirrors real-world situations while protecting patient privacy—participants can build and improve models that more effectively consider diverse backgrounds and conditions.

You’re challenged to develop advanced predictive models for allogeneic HCT that enhance both accuracy and fairness in survival predictions. The goal is to address disparities by bridging diverse data sources, refining algorithms, and reducing biases to ensure equitable outcomes for patients across diverse race groups. Your work will help create a more just and effective healthcare environment, ensuring every patient receives the care they deserve.        

# Evaluation
## Evaluation Criteria
The evaluation of prediction accuracy in the competition will involve a specialized metric known as the Stratified Concordance Index (C-index), adapted to consider different racial groups independently. This method allows us to gauge the predictive performance of models in a way that emphasizes equitability across diverse patient populations, particularly focusing on racial disparities in transplant outcomes.

## Concordance index
It represents the global assessment of the model discrimination power: this is the model’s ability to correctly provide a reliable ranking of the survival times based on the individual risk scores.

The concordance index is a value between 0 and 1 where:

* 0.5 is the expected result from random predictions,
* 1.0 is a perfect concordance and,
* 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)


# Rank


# Model Performances
| **Model Name**                         | **Public Score**      | **Private Score** |
|:--------------------------------------:|:---------------------:|:-----------------:|
| Baseline XGB                           |                |NA          |


#### Competetion
@misc{equity-post-HCT-survival-predictions,
    author = {Tushar Deshpande and Deniz Akdemir and Walter Reade and Ashley Chow and Maggie Demkin and Yung-Tsi Bolon},
    title = {CIBMTR - Equity in post-HCT Survival Predictions},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/equity-post-HCT-survival-predictions}},
    note = {Kaggle}
}