# TextClassification
## This repository contains code for text classification using Keras on top of TensorFlow 
Steps:
------
1. Read two input files - impact.csv and part.csv
2. Merge them to create a single dataset that contains part number, part description and other part features along with the aircraft model that gets impacted
3. Convert categorical features to integer
4. Merge two text columns into one that will be further used for BOW features
5. Get BOW features - limit to num_row, MAX_BOW_FEA, keepNum=True, hadleAbbr=True, ngram[1,2]
6. Prepare modeling dataset by keeping only relevant features and the target model number (BOW + column features + ModelNO)
7. Use One Hot encoding for Target 
8. Use 80% for training and 20% for testing (simple split)
9. Build a multiple layer perceptron with 3 layers, with 2 hidden layers of RELU and one output layer with softmax and 50% deopout (just used some default architecture)
10. Run for 20 epochs with batch_size=128. Make sure to save logfile for TensorBoard visualization
11. Finally print out the Final Validation Score
