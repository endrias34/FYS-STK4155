# FYS-STK4155 project 3 
In this project, we integrate the Convolutional Neural Networks (CNNs) and Strengthen-Operate-Subtract (SOS) boosting algorithm to denoise a microseismic dataset. 

All the analysis presented in our report can be reproduced by using our python codes and the corresponding Jupyther notebook.

## Folder structure 
Report   ---> Contains our report file (PDF and Latex)

plot_img ---> Contain all the results of project-3 (Figures)

src      ---> Contain all the python codes and the corresponding Jupyther notebooks 

model    ---> Contain all the trained models 

weights  ---> Contain all the trained weights

data     ---> Contain some of the test data sets

### Unit test codes: 

test_MultiClass.py

test_NN_regression.py

test_linear_regression.py



## Running the codes
The Jupyther notebooks contain all the analysis we did in project-3 and they can be run cell by cell

Note that to do the grid search of the CNN hyperparamers is very computionally heavy. 

For the readers that do not have powerful GPU to run the code, the trained models and weights from this project are provided in folders 'model' and 'weights', respectively.

The unit tests can be run by writting the following command: 
```
pytest -v
```

## Authors
Jing Sun and
Endrias Getachew Asgedom

