##############################
#                            #
# Weather Data Set - Final   #
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
set.seed(2020)
use_session_with_seed(
  2020,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Loading data
data("daily")

# Obtaining response
total_prec = range_01(apply(daily$precav, 2, sum))

# Creating functional data
temp_data = array(dim = c(65, 35, 1))
tempbasis65  = create.fourier.basis(c(0,365), 65)
timepts = seq(1, 365, 1)
temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)

# Changing into fdata
weather_fdata = fdata(daily$tempav, argvals = 1:365, rangeval = c(1, 365))

# Data set up
temp_data[,,1] = temp_fd$coefs

# Choosing fold number
num_folds = 35

# Creating folds
fold_ind = createFolds(total_prec, k = num_folds)

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
  train_x = weather_fdata[-fold_ind[[i]],]
  test_x = weather_fdata[fold_ind[[i]],]
  train_y = total_prec[-fold_ind[[i]]]
  test_y = total_prec[fold_ind[[i]]]
  
  # Setting up for FNN
  weather_data_train <- array(dim = c(65, ncol(temp_data) - length(fold_ind[[i]]), 1))
  weather_data_test <- array(dim = c(65, length(fold_ind[[i]]), 1))
  weather_data_train[,,1] = temp_data[, -fold_ind[[i]], ]
  weather_data_test[,,1] = temp_data[, fold_ind[[i]], ]
  
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
  func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda=0:5, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Running FNN for weather
  fnn_example = FNN(resp = train_y, 
                        func_cov = weather_data_train, 
                        scalar_cov = NULL,
                        basis_choice = c("fourier"), 
                        num_basis = 5,
                        hidden_layers = 2,
                        neurons_per_layer = c(16, 8),
                        activations_in_layers = c("relu", "sigmoid"),
                        domain_range = list(c(1, 365)),
                        epochs = 250,
                        output_size = 1,
                        loss_choice = "mse",
                        metric_choice = list("mean_squared_error"),
                        val_split = 0.2,
                        patience_param = 25,
                        learn_rate = 0.05,
                        early_stop = T,
                        print_info = F)
  
  # Predicting using FNN for weather
  pred_fnn = FNN_Predict(fnn_example,
                         weather_data_test, 
                         scalar_cov = NULL,
                         basis_choice = c("fourier"), 
                         num_basis = c(5),
                         domain_range = list(c(1, 365)))
  
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
  error_mat_lm[i, 2] = c(pred_basis)
  error_mat_pc1[i, 2] = pred_pc
  error_mat_pc2[i, 2] = pred_pc2
  error_mat_pc3[i, 2] = pred_pc3
  error_mat_pls1[i, 2] = pred_pls
  error_mat_pls2[i, 2] = pred_pls2
  error_mat_np[i, 2] = pred_np
  error_mat_fnn[i, 2] = pred_fnn
 
  # Printing iteration number
  print(paste0("Done Iteration: ", i))
  
}

# Initializing final table: average of errors
Final_Table_Weather = matrix(nrow = 8, ncol = 2)

# Collecting errors
Final_Table_Weather[1, 1] = mean(error_mat_lm[,1], na.rm = T)
Final_Table_Weather[2, 1] = mean(error_mat_np[,1], na.rm = T)
Final_Table_Weather[3, 1] = mean(error_mat_pc1[,1], na.rm = T)
Final_Table_Weather[4, 1] = mean(error_mat_pc2[,1], na.rm = T)
Final_Table_Weather[5, 1] = mean(error_mat_pc3[,1], na.rm = T)
Final_Table_Weather[6, 1] = mean(error_mat_pls1[,1], na.rm = T)
Final_Table_Weather[7, 1] = mean(error_mat_pls2[,1], na.rm = T)
Final_Table_Weather[8, 1] = mean(error_mat_fnn[,1], na.rm = T)

# R_Squared value
Final_Table_Weather[1, 2] = rsq(c(error_mat_lm[,2]), total_prec)
Final_Table_Weather[2, 2] = rsq(error_mat_np[,2], total_prec)
Final_Table_Weather[3, 2] = rsq(error_mat_pc1[,2], total_prec)
Final_Table_Weather[4, 2] = rsq(error_mat_pc2[,2], total_prec)
Final_Table_Weather[5, 2] = rsq(error_mat_pc3[,2], total_prec)
Final_Table_Weather[6, 2] = rsq(error_mat_pls1[,2], total_prec)
Final_Table_Weather[7, 2] = rsq(error_mat_pls2[,2], total_prec)
Final_Table_Weather[8, 2] = rsq(error_mat_fnn[,2], total_prec)

# Looking at results
Final_Table_Weather

#########################
# Usual Neural Networks #
#########################

# Setting seed
set.seed(2020)

# Library
library(nnet)

# Converting data
sim_df <- as.data.frame(t(daily$tempav))

# y values
response <- total_prec

# Putting together
final_df <- cbind(sim_df, response)

# Initializing
MSPE_nn <- c()
pred_nn <- c()

# Looping
for (u in 1:35) {
  
  #u = 1
  
  # Set
  set <- u
  
  # Creating set 1 and 2
  train_1 <- final_df[-set,]
  train_2 <- setdiff(final_df, train_1)
  
  # Scaling
  x.1.unscaled <- train_1[,1:365]
  y.1 <- train_1[,366]
  x.2.unscaled <- train_2[,1:365]
  y.2 <- train_2[,366]
  
  x.1 <- rescale(x.1.unscaled, x.1.unscaled)
  x.2 <- rescale(x.2.unscaled, x.1.unscaled)
  
  # Creating data frame of tuning parameters
  tuning_par <- expand.grid(c(3, 5), c(0.01, 0.1))
  colnames(tuning_par) <- c("Nodes", "Decay")
  
  # Running through apply
  results_nn <- apply(tuning_par, 1, function(x){
    
    # Running neural network
    MSE.final <- 9e99
    for(i in 1:1){
      nn <- nnet(y = y.1, x = x.1, 
                 linout = TRUE, 
                 size = x[1], 
                 decay = x[2], 
                 maxit = 500, 
                 trace = FALSE,
                 MaxNWts = 5000)
      MSE <- nn$value/nrow(x.1)
      if(MSE < MSE.final){ 
        MSE.final <- MSE
        nn.final <- nn
      }
      #    check <- c(check,MSE.final)
    }
    
    # Getting Errors
    MSE <- nn.final$value/nrow(x.1)
    MSPE <- mean((y.2 - predict(nn.final, x.2))^2)
    pred <- predict(nn.final, x.2)[1,]
    
    # Putting together
    df_returned <- data.frame(Nodes = x[1], Decay = x[2], MSE = MSE, MSPE = MSPE, Predict_Value = pred)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  q1_errors <- do.call(rbind, results_nn)
  
  # Saving the predictions
  pred_nn[u] <- q1_errors[which.min(q1_errors$MSPE), 5]
  
  print(u)
  
}

# mean
mspe_nn = mean((pred_nn - response)^2)
mspe_nn

# rsquared
rsq(pred_nn, response)


# Check 1
# Check 2