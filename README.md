# MCI-risk-prediction-model
1. Our data was kindly provided by the Shenzhen Chronic Disease Control and Prevention Center.  2. In terms of feature selection, we came up with a nifty integrated method to select the best features for our analysis.  3. As for our modeling approach, we developed a semi-supervised learning method using a pseudo-labeling technique with replacement.


# function of Model_feature_selection.py
Input: Training set X and Y.
Output: Subset of features after selecting feature selection method (>2) and training 10 times using different random seeds (=10).

Processing steps:
1. Perform feature selection using "pearson" correlation coefficient function and filter method (a single feature selection method).
2. Perform feature selection using "chi2" function and filter method.
3. Perform feature selection using "REF" function and wrapper method (a single feature selection method).
4. Perform feature selection using "lgbm" function and embedding method (a single feature selection method).
5. Perform feature selection using "et" function and embedding method.
6. Perform ensemble method using "process" function by iterating through the 5 methods and voting for the feature subset that occurs more than 2 times.
7. In the "main" function, iteratively run the model 10 times using different random seed and voting for feature subset that occurs consistently across all 10 iterations.

# function of Model_Pseodo-baded semi-ML.py

import Model_feature_selection
from Model_feature_selection import PseudoTotal
assess=PseudoTotal(fp=filepath,max_iter=9,basic_model=basic_model,n_top=50,n_bottom=50,featnum_criteria=30,feature_traNum=0,put_back=False,model_use_Augment=True).main()

The input parameters for the model are:
- "fp": the filepath where the data is located.
- "max_iter": the maximum number of self-training iterations.
- "basic_model": the base model to which the pseudo-labels are added for training.
- "n_top" and "n_bottom": the number of instances to be labeled with high-confidence predictions and low-confidence predictions, respectively.
- "feature_traNum": the maximum number of times a feature can participate in training. Default is 0, meaning that a feature is only used once in supervised training.
- "featnum_criteria": the number of features obtained from individual feature selection methods. The default is 30.
- "put_back": whether to put the unlabeled data back in after applying data augmentation.
- "model_use_Augment": whether to use augmented data during model training. If True, the model is unsupervised.
