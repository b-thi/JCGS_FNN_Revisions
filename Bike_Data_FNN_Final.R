##############################
#                            #
# Bike Data Set - Final      #
#                            #
##############################

# Functional Neural Networks

# Libraries
library(fda.usc)
source("FNN.R")

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1995)
use_session_with_seed(
  1995,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Loading data
load("bike.RData")

# Obtaining response
rentals = log10(bike$y)

# define the time points on which the functional predictor is observed. 
timepts = bike$timepts

# define the fourier basis 
nbasis = 31
spline_basis = create.fourier.basis(c(1,24), nbasis)

# convert the functional predictor into a fda object
bike_fd =  Data2fd(timepts, t(bike$temp), spline_basis)
bike_deriv1 = deriv.fd(bike_fd)
bike_deriv2 = deriv.fd(bike_deriv1)

# Testing with bike data
func_cov_1 = bike_fd$coefs
#func_cov_2 = bike_deriv1$coefs
#func_cov_3 = bike_deriv2$coefs
bike_data = array(dim = c(31, 102, 1))
bike_data[,,1] = func_cov_1
#bike_data[,,2] = func_cov_2
#bike_data[,,3] = func_cov_3

# fData Object
bike_fdata = fdata(bike$temp, argvals = 1:24, rangeval = c(1, 24))

# Choosing fold number
num_folds = 10

# Creating folds
fold_ind = createFolds(rentals, k = num_folds)

# Initializing matrices for results
error_mat_lm = matrix(nrow = num_folds, ncol = 2)
error_mat_pc1 = matrix(nrow = num_folds, ncol = 2)
error_mat_pc2 = matrix(nrow = num_folds, ncol = 2)
error_mat_pc3 = matrix(nrow = num_folds, ncol = 2)
error_mat_pls1 = matrix(nrow = num_folds, ncol = 2)
error_mat_pls2 = matrix(nrow = num_folds, ncol = 2)
error_mat_np = matrix(nrow = num_folds, ncol = 2)
error_mat_fnn = matrix(nrow = num_folds, ncol = 2)

# Looping to get results
for (i in 1:num_folds) {
  
  ################## 
  # Splitting data #
  ##################

  # Test and train
  train_x = bike_fdata[-fold_ind[[i]],]
  test_x = bike_fdata[fold_ind[[i]],]
  train_y = rentals[-fold_ind[[i]]]
  test_y = rentals[fold_ind[[i]]]
  
  # Setting up for FNN
  bike_data_train = array(dim = c(31, nrow(train_x$data), 1))
  bike_data_test = array(dim = c(31, nrow(test_x$data), 1))
  bike_data_train[,,1] = bike_data[, -fold_ind[[i]], ]
  bike_data_test[,,1] = bike_data[, fold_ind[[i]], ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional Linear Model (Basis)
  l=2^(-4:10)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = predict(func_basis[[1]], test_x)
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 8)
  pred_pc = predict(func_pc$fregre.pc, test_x)
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:6)
  pred_pls = predict(func_pls, test_x)
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Running FNN for bike
  bike_example <- FNN(resp = train_y, 
                      func_cov = bike_data_train, 
                      scalar_cov = NULL,
                      basis_choice = c("fourier"), 
                      num_basis = c(3),
                      hidden_layers = 4,
                      neurons_per_layer = c(32, 32, 32, 32),
                      activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                      domain_range = list(c(1, 24)),
                      epochs = 500,
                      output_size = 1,
                      loss_choice = "mse",
                      metric_choice = list("mean_squared_error"),
                      val_split = 0.15,
                      learn_rate = 0.002,
                      patience_param = 15,
                      early_stop = T,
                      print_info = F)
  
  # Predicting
  pred_fnn = FNN_Predict(bike_example,
                          bike_data_test, 
                          scalar_cov = NULL,
                          basis_choice = c("fourier"), 
                          num_basis = c(3),
                          domain_range = list(c(1, 24)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm[i, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1[i, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2[i, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3[i, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1[i, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2[i, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np[i, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_fnn[i, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  error_mat_lm[i, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_pc1[i, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_pc2[i, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_pc3[i, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_pls1[i, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_pls2[i, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_np[i, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_fnn[i, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", i))
  
}

# Initializing final table: average of errors
Final_Table_Bike = matrix(nrow = 8, ncol = 2)

# Collecting errors
Final_Table_Bike[1, ] = colMeans(error_mat_lm, na.rm = T)
Final_Table_Bike[2, ] = colMeans(error_mat_np, na.rm = T)
Final_Table_Bike[3, ] = colMeans(error_mat_pc1, na.rm = T)
Final_Table_Bike[4, ] = colMeans(error_mat_pc2, na.rm = T)
Final_Table_Bike[5, ] = colMeans(error_mat_pc3, na.rm = T)
Final_Table_Bike[6, ] = colMeans(error_mat_pls1, na.rm = T)
Final_Table_Bike[7, ] = colMeans(error_mat_pls2, na.rm = T)
Final_Table_Bike[8, ] = colMeans(error_mat_fnn, na.rm = T)

# Looking at results
Final_Table_Bike

# Check 1
# Check 2