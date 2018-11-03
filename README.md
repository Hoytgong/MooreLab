# Steps Taken

Dataset used: 29 SNPs of breast cancer cases & controls (2504 participants)

1. In TPOT Breast Cancer Data:
	A) Split dataset into test data, validate data, precision_medicine_data
		1) precision_medicine_data is balanced set of case & controls (n = 100)
			a. Exported as 'premed.csv'
		2) The precision_medicine_data is then dropped from original dataset
			a. Exported as 'test_validate_data.csv'
	B) Create TPOT classifer to run on test_validate_data 
		1) Fitted on TRAINING X (all the features) & Y (phenotype column)
		2) Found CV score using now fitted tpot on TEST X & Y
			a. Exported pipeline as 'bcdata_pipeline.py'

2. In PythonBreastCancer