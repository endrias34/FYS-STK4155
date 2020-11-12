
## FYS-STK4155 project 2 
We investigate regression and classification problems using a noise contaminated Franke function and the MNIST dataset. To solve the regression problem, we use a fully connected neural network (NN) and Stochastic Gradient Descent (SGD)-based Ordinary Least Square (OLS) and Ridge regression methods.

Our report can be read at report2_fysstk_magnubgr_michaesb.pdf and this is where we present our findings.

Here we have made two packages; one called Regression_package and one called NeuralNet_package. The Regression_package contains Linear Regression which has 3 methods (OLS, Ridge and Lasso) and LogisticRegressor which uses a gradient descent to minimize the loss function. The Linear Regression methods are taking from a previous project.

The NeuralNet_package contains a classifying NeuralNet and regression NeuralNet. NeuralNet_package is designed to be very similiar to how you would use Scikit-Learns package on NeuralNet.

Running the scripts
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

Authors
Michael Bitney
Magnus Grøndalen
