##############################
#                            #
# Tecator Data Set - Final   #
#                            #
##############################

# Functional Neural Networks

# Libraries
library(fda.usc)

# Setting up environment
library(reticulate)
use_condaenv(condaenv = 'PFDA', conda = "C:/Users/Barinder/anaconda3/envs/Python37/python.exe")
use_python("C:/Users/Barinder/anaconda3/envs/Python37/python.exe")

# Dataset
# MAKE SURE YOU LOAD FDA.USC PACKAGE BEFORE YOU LOAD FDA PACKAGE IN ORDER TO GET
# THE PROPER LOAD IN OF THE DATA
data("tecator")

# Source for FNN
source("FNN.R")

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(17)
use_session_with_seed(
  17,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# define the time points on which the functional predictor is observed. 
timepts = tecator$absorp.fdata$argvals

# define the fourier basis 
nbasis = 29
spline_basis = create.fourier.basis(tecator$absorp.fdata$rangeval, nbasis)

# convert the functional predictor into a fda object and getting deriv
tecator_fd =  Data2fd(timepts, t(tecator$absorp.fdata$data), spline_basis)
tecator_deriv = deriv.fd(tecator_fd)
tecator_deriv2 = deriv.fd(tecator_deriv)

# Non functional covariate
tecator_scalar = data.frame(water = tecator$y$Water)

# Response
tecator_resp = tecator$y$Fat

# Getting data into right format
tecator_data = array(dim = c(nbasis, 215, 1))
tecator_data[,,1] = tecator_deriv2$coefs

# Splitting into test and train
ind = 1:165
tec_data_train <- array(dim = c(nbasis, length(ind), 1))
tec_data_test <- array(dim = c(nbasis, nrow(tecator$absorp.fdata$data) - length(ind), 1))
tec_data_train[,,1] = tecator_data[, ind, ]
tec_data_test[,,1] = tecator_data[, -ind, ]
tecResp_train = tecator_resp[ind]
tecResp_test = tecator_resp[-ind]
scalar_train = data.frame(tecator_scalar[ind,1])
scalar_test = data.frame(tecator_scalar[-ind,1])

# Setting up network
tecator_comp = FNN(resp = tecResp_train, 
                      func_cov = tec_data_train, 
                      scalar_cov = scalar_train,
                      basis_choice = c("fourier"), 
                      num_basis = 3,
                      hidden_layers = 6,
                      neurons_per_layer = c(24, 24, 24, 24, 24, 58),
                      activations_in_layers = c("relu", "relu", "relu", "relu", "relu", "linear"),
                      domain_range = list(c(850, 1050)),
                      epochs = 300,
                      output_size = 1,
                      loss_choice = "mse",
                      metric_choice = list("mean_squared_error"),
                      val_split = 0.15,
                      patience_param = 35,
                      learn_rate = 0.005,
                      decay_rate = 0,
                      batch_size = 32,
                      early_stop = F,
                      print_info = T)

# Predicting
pred_tec = FNN_Predict(tecator_comp,
                       tec_data_test, 
                       scalar_cov = scalar_test,
                       basis_choice = c("fourier"), 
                       num_basis = 3,
                       domain_range = list(c(850, 1050)))

# Getting back results
MEP = mean(((pred_tec - tecResp_test)^2))/var(tecResp_test)
Rsquared = sum((pred_tec - mean(tecResp_test))^2)/sum((tecResp_test - mean(tecResp_test))^2)

# Check 1
# Check 2