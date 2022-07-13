<h1>Causal Machine Learning</h1> 

This is the summary of my project as a Faculty Data Science Fellow. The aim of this project is to find the best treatment option for indivual customers/patients/... using a randomised, controlled trail dataset.

I used the open source marketing data called Hillstrom. The data consists of a customer base of the last 12 month and whether they bought merchandise for women or men. Interventions are randomly distributed amongst the customer base (randomised control trial). There are three different interventions:

- Email with merchandise for women
- Email with merchandise for men
- No email

Within two weeks after the email intervention, we know whether customers visited the website and purchased an item. In this project, it will be rated as an success if a costomer visits the website within the two weeks after the intervention.

I use three different learners to compute the best treatment option: S-, T-, and correlated ST-learner. The former two learners will be used with three different classifier, namely, linear regression, random forest, and xgboost. The latter uses two xgboost models.

> The files:

* __CausalML.ipynb__: uses all other files to compute the best option with each learner and compares their performance. A detailed description of the learners and the data set

* __class_learners.py__: the classes for each of the three learners and an additional "learner" class, which operates as a parent class to each of the three learner classes

* __metric.py__: class for the ERUPT metric

* __feature_engineering.py__: functions for feature engineering of the trainings data

* __plots.py__: functions for the plots used in __CausalML.ipynb__



