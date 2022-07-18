<h1>Causal Machine Learning</h1> 

The aim of this project is to find the best treatment option for indivual customers/patients/... given a randomised, controlled trail dataset with multiple treatment options.

I used the open source marketing data called [Hillstrom](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html). The data consists of a customer base of the last 12 month and whether they bought merchandise for women or men. Interventions are randomly distributed amongst the customer base (randomised control trial). There are three different interventions:

- Email with merchandise for women
- Email with merchandise for men
- No email

Within two weeks after the email intervention, we know whether customers visited the website and purchased an item. In this project, it will be rated as an success if a costomer visits the website up to two weeks after the intervention.

I use three different learners to compute the best treatment option: 

* _S-Leaner_

* _T-Learner_

* _correlated ST-learner_ 

The former two learners will be used with three different classifier:

- linear regression 
- random forest
-  xgboost

The latter uses two xgboost models.
<br>
<br>
> ## The files:
<br>

* __CausalML.ipynb__: uses all other files to compute the best option with each learner and compares their performance. A detailed description of the learners and the data set

* __class_learners.py__: the classes for each of the three learners and an additional "learner" class, which operates as a parent class to each of the three learner classes

* __metric.py__: class for the ERUPT metric

* __feature_engineering.py__: functions for feature engineering of the trainings data

* __plots.py__: functions for the plots used in __CausalML.ipynb__

<br>
<br>

>  ## The outcome:

<br>

To evaluate the outcomes, I use the ERUPT [metric](https://medium.com/building-ibotta/erupt-expected-response-under-proposed-treatments-ff7dd45c84b4). (A detailed description of the metric can be found in **CausalML.ipynb**.) All S-learner and T-learner with random forest perform best. However, there are a few issues with the data that this result reflects: even though it is a reasonable sized dataset, only about 15% of the customers visit the store within the two weeks after the intervention. Of these 15%, the second intervention ("Mens E-mail") is the strongest intervention. Therefore, a learner that proposes only "Mens E-mail" fares fairly well. (Detailed analysis in **CausalML.ipynb**)
