# Steps Taken

Dataset used: 29 SNPs of breast cancer cases & controls (2504 participants)

1. In TPOT Breast Cancer Data:
   - A) Split dataset into test data, validate data, precision_medicine_data
     - 1) precision_medicine_data is balanced set of case & controls (n = 100)
	   - Exported as 'premed.csv'
	 - 2) The precision_medicine_data is then dropped from original dataset
	   - Exported as 'test_validate_data.csv'
   - B) Create TPOT classifer to run on test_validate_data 
	 - 1) Fitted on TRAINING X (all the features) & Y (phenotype column)
	 - 2) Found CV score using now fitted tpot on TEST X & Y
	   - Exported pipeline as 'bcdata_pipeline.py'

2. In PythonBreastCancer
   - A) Take the exported pipeline and train it again on the test_validate_data 
     - 1) Call predict method of exported_pipeline on testing_features to get predicted phenotypes that we compare to testing_target
       - That error sums to 1
   - B) Probability machine approach with predict_proba method
     - 1) Take array of probability that each participant has phenotype 1
       - Visualize density plot of test_validate_data features' probabilities
       - Visualize density plot of pm_features 
   - C) Change individual genotypes of the patient 