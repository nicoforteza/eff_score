# Replication code

This repository provides the code for replicating the package of the working paper ["A Score Function to Prioritize Editing in Household Survey Data: A Machine Learning Approach"](https://www.bde.es/wbe/es/publicaciones/analisis-economico-investigacion/documentos-trabajo/a-score-function-to-prioritize-editing-in-household-survey-data--a-machine-learning-approach.html), by Nicolás Forteza and Sandra García-Uribe.
The file ```requirements.txt``` provides the list of packages needed for executing the code. It is divided in two parts:

 - The file ```train.py``` provides the script for training the models. The input is held in ```data``` in ```.csv``` format. 
Note that a specific function for reading and writing data is provided, so that each feature or covariate is loaded with the desired format.
 - The notebook ```paper.ipynb``` provides the evaluation, threshold selection and interpretability part of the draft.

The data files are emtpy for confidentiality issues, although we're working on providing a synthetic dataset to execute both the training script and the notebook with the results. The training script was executed in a 32 multi-core 3.4GHz AMD Ryzen, with 128Gb of RAM.   

