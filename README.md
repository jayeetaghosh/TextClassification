# TextClassification
## This repository contains code for text classification using Keras on top of TensorFlow 
Steps:
------
1. Read two input files - impact.csv and part.csv
2. Merge them to create a single modeling dataset that contains part number, part description and other part features along with the aircraft model that gets impacted
3. Convert categorical features to integer
4. Merge two text columns into one that will be further used for BOW features
5. Get BOW features - limit to num_row, MAX_BOW_FEA, keepNum=True, hadleAbbr=True, ngram[1,2]
