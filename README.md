# LendingClubLoanDataset
This repo contains ML practice on Lending Club Personal Loan dataset.  

The dataset is obtained from Lending Club, contains complete loan data for all loans issued through 2007-2015, including current loan status and all other related information, e.g., applicants' information, payment, etc,. The file consists of 890K rows and 75 variables.   

The current loan status fall in: default, charge-off, fully paid, late, etc,. To define the learning objective, we subset the data by choosing current loan status of 'default', 'charge-off' and 'full paid' only, then encode the first two as 'bad loans' and third as 'good loans'. Now, the goal of this practice is to build classifier that could predict propensity of 'bad loans' for current clients, who are still in the process of paying the loans, so that the loan investors might take actions to prevent occurance of 'bad loans' in the end.  

First of all, we keep 10% of all the data in a stratified way, which meant to keep targe class ratio constant, as hold out set for validation. After initial feature screening, 20 of 75 have over 70% missing values, these were removed directly. For couple of categorical features, combine minory categories as 'other category' after checking their distributions. Also we create few new features based on information extracted from original ones. After all these pre-screening of features, we built pipelines with preprocessing steps, over sampling step, feature importance test step and classification step. Algorithms employed here are: Logistic regression, Support vector machine, Random forest classifier and Gradient boosting machine.  

After a combination of grid search and bayesian optimization for the hyperparameters and model parameter fine tune, we got several classifiers with different performance. To combine the power of them, we proceed with a stacking mechanism, which trained a logistic regression with the predictions of each classifier as feature. 
