##################################################
#############  Simulation Study    ############
#### Functional Neural Networks ##################
##################################################

###### MAIN CODE #######

##### Libraries #####
library(fda)
library(fda.usc)
library(tidyverse)
library(gridExtra)
library(ggpubr)
library(reshape)
source("FNN.R")

#############################################################
# 1 - Identity 
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1)
use_session_with_seed(
  1,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Number of sims
sim_num = 5

# Initializing matrices for results
error_mat_lm_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_1 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_1 = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################

  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = composite_approximator_other(response_func1,
                                        a = 0,
                                        b = 1,
                                        n = 500,
                                        x_obs = simSmooth$fd$coefs[,i], 
                                        beta = beta_coef[1]) +
      composite_approximator_other(response_func2,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[2]) +
      composite_approximator_other(response_func3,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[3]) +
      composite_approximator_other(response_func4,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[4]) +
      composite_approximator_other(response_func5,
                                   a = 0,
                                   b = 1,
                                   n = 500,
                                   x_obs = simSmooth$fd$coefs[,i], 
                                   beta = beta_coef[5]) +
      alpha[i]
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  # Setting up index
  ind = sample(1:300, 50)

  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
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
  
  # Running FNN for simulation
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 3,
                neurons_per_layer = c(16, 16, 16),
                activations_in_layers = c("relu", "linear", "linear"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_1[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_1[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_1[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_1[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_1[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_1[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_1[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_fnn_1[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  # error_mat_lm[u, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc1[u, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc2[u, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc3[u, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls1[u, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls2[u, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_np[u, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_fnn[u, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim1 = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim1[1, ] = colMeans(error_mat_lm_1, na.rm = T)
Final_Table_Sim1[2, ] = colMeans(error_mat_np_1, na.rm = T)
Final_Table_Sim1[3, ] = colMeans(error_mat_pc1_1, na.rm = T)
Final_Table_Sim1[4, ] = colMeans(error_mat_pc2_1, na.rm = T)
Final_Table_Sim1[5, ] = colMeans(error_mat_pc3_1, na.rm = T)
Final_Table_Sim1[6, ] = colMeans(error_mat_pls1_1, na.rm = T)
Final_Table_Sim1[7, ] = colMeans(error_mat_pls2_1, na.rm = T)
Final_Table_Sim1[8, ] = colMeans(error_mat_fnn_1, na.rm = T)

# Looking at results
Final_Table_Sim1


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#############################################################
# 2 - Exponential
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(2)
use_session_with_seed(
  2,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Initializing matrices for results
error_mat_lm_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_2 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_2 = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = exp(composite_approximator_other(response_func1,
                                            a = 0,
                                            b = 1,
                                            n = 500,
                                            x_obs = simSmooth$fd$coefs[,i], 
                                            beta = beta_coef[1]) +
                 composite_approximator_other(response_func2,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[2]) +
                 composite_approximator_other(response_func3,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[3]) +
                 composite_approximator_other(response_func4,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[4]) +
                 composite_approximator_other(response_func5,
                                              a = 0,
                                              b = 1,
                                              n = 500,
                                              x_obs = simSmooth$fd$coefs[,i], 
                                              beta = beta_coef[5]) +
                 alpha[i])
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  ystar = c(scale(ystar))
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
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
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 3,
                neurons_per_layer = c(16, 16, 16),
                activations_in_layers = c("relu", "linear", "linear"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_2[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_2[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_2[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_2[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_2[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_2[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_2[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_fnn_2[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  # error_mat_lm[u, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc1[u, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc2[u, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc3[u, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls1[u, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls2[u, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_np[u, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_fnn[u, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim2 = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim2[1, ] = colMeans(error_mat_lm_2, na.rm = T)
Final_Table_Sim2[2, ] = colMeans(error_mat_np_2, na.rm = T)
Final_Table_Sim2[3, ] = colMeans(error_mat_pc1_2, na.rm = T)
Final_Table_Sim2[4, ] = colMeans(error_mat_pc2_2, na.rm = T)
Final_Table_Sim2[5, ] = colMeans(error_mat_pc3_2, na.rm = T)
Final_Table_Sim2[6, ] = colMeans(error_mat_pls1_2, na.rm = T)
Final_Table_Sim2[7, ] = colMeans(error_mat_pls2_2, na.rm = T)
Final_Table_Sim2[8, ] = colMeans(error_mat_fnn_2, na.rm = T)

# Looking at results
Final_Table_Sim2


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#############################################################
# 3 - Sigmoidal
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(3)
use_session_with_seed(
  3,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Initializing matrices for results
error_mat_lm_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_3 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_3 = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = 1/(1 + exp(-(composite_approximator_other(response_func1,
                                                     a = 0,
                                                     b = 1,
                                                     n = 500,
                                                     x_obs = simSmooth$fd$coefs[,i], 
                                                     beta = beta_coef[1]) +
                          composite_approximator_other(response_func2,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[2]) +
                          composite_approximator_other(response_func3,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[3]) +
                          composite_approximator_other(response_func4,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[4]) +
                          composite_approximator_other(response_func5,
                                                       a = 0,
                                                       b = 1,
                                                       n = 500,
                                                       x_obs = simSmooth$fd$coefs[,i], 
                                                       beta = beta_coef[5])) +
                        alpha[i]))
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
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
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 1,
                neurons_per_layer = c(16),
                activations_in_layers = c("sigmoid"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  error_mat_lm_3[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_3[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_3[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_3[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_3[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_3[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_3[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_fnn_3[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  # error_mat_lm[u, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc1[u, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc2[u, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc3[u, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls1[u, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls2[u, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_np[u, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_fnn[u, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim3 = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim3[1, ] = colMeans(error_mat_lm_3, na.rm = T)
Final_Table_Sim3[2, ] = colMeans(error_mat_np_3, na.rm = T)
Final_Table_Sim3[3, ] = colMeans(error_mat_pc1_3, na.rm = T)
Final_Table_Sim3[4, ] = colMeans(error_mat_pc2_3, na.rm = T)
Final_Table_Sim3[5, ] = colMeans(error_mat_pc3_3, na.rm = T)
Final_Table_Sim3[6, ] = colMeans(error_mat_pls1_3, na.rm = T)
Final_Table_Sim3[7, ] = colMeans(error_mat_pls2_3, na.rm = T)
Final_Table_Sim3[8, ] = colMeans(error_mat_fnn_3, na.rm = T)

# Looking at results
Final_Table_Sim3

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#############################################################
# 4 - Log
#############################################################

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(4)
use_session_with_seed(
  4,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Beta(s) function
beta_coef = runif(5, min = 0, max = 2)

# Initializing matrices for results
error_mat_lm_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc1_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc2_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pc3_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls1_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_pls2_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_np_4 = matrix(nrow = sim_num, ncol = 1)
error_mat_fnn_4 = matrix(nrow = sim_num, ncol = 1)

# Looping to get results
for (u in 1:sim_num) {
  
  ################## 
  # Splitting data #
  ##################
  
  # Generating data for each of observations
  sim_data <- matrix(nrow = 300, ncol = 100)
  for (j in 1:300) {
    const = rnorm(1,)
    a = rnorm(1)
    for (i in 1:100) {
      c = rnorm(1, i/100)
      sim_data[j, i] <- c*exp(a) + sin(a) + const
    }
  }
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # INTEGRATION FUNCTIONS
  response_func1 = function(x, x_obs, beta){
    result = beta*1*(x_obs[1] + 
                       x_obs[2]*sin(2*pi*x/1) + 
                       x_obs[3]*cos(2*pi*x/1) + 
                       x_obs[4]*sin(2*2*pi*x/1) + 
                       x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func2 = function(x, x_obs, beta){
    result = beta*sin(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func3 = function(x, x_obs, beta){
    result = beta*cos(2*pi*x/1)*(x_obs[1] + 
                                   x_obs[2]*sin(2*pi*x/1) + 
                                   x_obs[3]*cos(2*pi*x/1) + 
                                   x_obs[4]*sin(2*2*pi*x/1) + 
                                   x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func4 = function(x, x_obs, beta){
    result = beta*sin(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  response_func5 = function(x, x_obs, beta){
    result = beta*cos(2*2*pi*x/1)*(x_obs[1] + 
                                     x_obs[2]*sin(2*pi*x/1) + 
                                     x_obs[3]*cos(2*pi*x/1) + 
                                     x_obs[4]*sin(2*2*pi*x/1) + 
                                     x_obs[5]*cos(2*2*pi*x/1))
  }
  
  
  # Generating alpha
  alpha = runif(300, 0, 1)
  
  # Getting y values
  y = c()
  for (i in 1:300) {
    y[i] = log(abs(composite_approximator_other(response_func1,
                                                a = 0,
                                                b = 1,
                                                n = 500,
                                                x_obs = simSmooth$fd$coefs[,i], 
                                                beta = beta_coef[1]) +
                     composite_approximator_other(response_func2,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[2]) +
                     composite_approximator_other(response_func3,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[3]) +
                     composite_approximator_other(response_func4,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[4]) +
                     composite_approximator_other(response_func5,
                                                  a = 0,
                                                  b = 1,
                                                  n = 500,
                                                  x_obs = simSmooth$fd$coefs[,i], 
                                                  beta = beta_coef[5])) +
                 alpha[i])
    
  }
  
  # Getting observed y values
  errors = rnorm(300, mean = 0, sd = 0.1)
  ystar = y + errors
  
  # Creating fourier basis
  sim_basis <- create.fourier.basis(c(0, 1), 5)
  sim_vals <- seq(0, 0.99, 0.01)
  simSmooth <- smooth.basis(sim_vals, 
                            t(sim_data), 
                            sim_basis)
  
  # Creating functional data object
  sim_fd <- Data2fd(sim_vals, t(sim_data), sim_basis)
  sim_fdata = fdata(sim_data, argvals = seq(0, 0.99, 0.01), rangeval = c(0, 1))

  # Setting up index
  ind = sample(1:300, 50)
  
  # Test and train
  train_x = sim_fdata[-ind,]
  test_x = sim_fdata[ind,]
  train_y = ystar[-ind]
  test_y = ystar[ind]
  
  # Setting up for FNN
  sim_data_fnn = array(dim = c(5, 300, 1))
  
  # Getting coefficients
  sim_data_fnn[,,1] = sim_fd$coefs
  
  sim_data_train <- array(dim = c(5, 250, 1))
  sim_data_test <- array(dim = c(5, 50, 1))
  
  sim_data_train[,,1] = sim_data_fnn[, -ind, ]
  sim_data_test[,,1] = sim_data_fnn[, ind, ]
  
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
  use_session_with_seed(
    u,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  fnn_sim = FNN(resp = train_y, 
                func_cov = sim_data_train, 
                scalar_cov = NULL,
                basis_choice = c("fourier"), 
                num_basis = 5,
                hidden_layers = 3,
                neurons_per_layer = c(32, 32, 32),
                activations_in_layers = c("sigmoid", "relu", "linear"),
                domain_range = list(c(0, 1)),
                epochs = 250,
                output_size = 1,
                loss_choice = "mse",
                metric_choice = list("mean_squared_error"),
                val_split = 0.2,
                patience_param = 25,
                learn_rate = 0.001,
                early_stop = T,
                print_info = F)
  
  # Predicting
  pred_fnn = FNN_Predict(fnn_sim,
                         sim_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(0, 1)))
  
  ###################
  # Storing Results #
  ###################
  
  # Standardizing means
  std_y = scale(test_y)
  
  # MSPE Results
  error_mat_lm_4[u, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  error_mat_pc1_4[u, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  error_mat_pc2_4[u, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  error_mat_pc3_4[u, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  error_mat_pls1_4[u, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  error_mat_pls2_4[u, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  error_mat_np_4[u, 1] = mean((pred_np - test_y)^2, na.rm = T)
  error_mat_fnn_4[u, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  # error_mat_lm[u, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc1[u, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc2[u, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc3[u, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls1[u, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls2[u, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_np[u, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_fnn[u, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim4 = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim4[1, ] = colMeans(error_mat_lm_4, na.rm = T)
Final_Table_Sim4[2, ] = colMeans(error_mat_np_4, na.rm = T)
Final_Table_Sim4[3, ] = colMeans(error_mat_pc1_4, na.rm = T)
Final_Table_Sim4[4, ] = colMeans(error_mat_pc2_4, na.rm = T)
Final_Table_Sim4[5, ] = colMeans(error_mat_pc3_4, na.rm = T)
Final_Table_Sim4[6, ] = colMeans(error_mat_pls1_4, na.rm = T)
Final_Table_Sim4[7, ] = colMeans(error_mat_pls2_4, na.rm = T)
Final_Table_Sim4[8, ] = colMeans(error_mat_fnn_4, na.rm = T)

# Looking at results
Final_Table_Sim4

# Check 1