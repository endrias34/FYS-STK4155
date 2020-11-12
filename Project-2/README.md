
# FYS-STK4155 project 2 
In this project, we investigate regression and classification problems. We use the Franke function and the MNIST as data sets. To solve the regression problem, we use a fully connected neural network (NN) and Stochastic Gradient Descent (SGD)-based Ordinary Least Square (OLS) and Ridge regression methods. The classification problem is solved using NNs and multinominal logistic regression.

All the analysis presented in our report can be reproduced by using our python codes and the corresponding Jupyther notebook.

## Folder structure 
Report  ---> Contains our report file (PDF and Latex)
Figures ---> Contain all the results of project-2 (Figures and Tabels)
src     ---> Contain all the python codes and the corresponding Jupyther notebooks 

## Running the codes
The following scripts can be run to test our different machine learning methods:

logreg_plots.py
logreg_comparing_performance_test.py
neuralnet_clf_performance_test.py
neuralnet_reg_comparison_sklearn.py
In order to run the python files you can simply type in the terminal:

python [NAME OF FILE].py
where the [NAME OF FILE] will replace the script you want to run.

If you want to run all of them and you have access to a terminal that can run a bash-script, you can do this by running:

./run_scripts/run_all.sh
We have a test file that runs multiple tests on the Regression_package and NeuralNet_package. You can run this by typing:

pytest -v
This will test both packages with test on for example R2-score, sigmoidfunction and so on.

## Authors
Jing Sun and
Endrias Getachew Asgedom
