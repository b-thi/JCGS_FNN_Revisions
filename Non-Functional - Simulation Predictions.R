##################################################
#############     Simulation Study    ############
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
library(tidyverse)
library(keras)
library(caret)
library(glmnet)
library(randomForest)
library(future.apply)
library(earth)
library(gam)
library(gbm)
source("FNN.R")

#############################################################
# 1 - Identity 
#############################################################

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
sim_num = 100

# Initializing matrices for results
error_mat_lm1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS1_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB1_nf = matrix(nrow = sim_num, ncol = 1)

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

  # Setting up index
  ind = sample(1:300, 50)
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
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
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                       newdata = test_f, 
                                       n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = TRUE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
  # Running FNN for bike (ONLY RUNNING THIS AGAIN TO MAKE SPLITS CONSISTENT)
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
  error_mat_lm1_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin1_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se1_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF1_nf[u, 1] = MSPE_rf
  error_mat_GBM1_nf[u, 1] = MSPE_gbm
  error_mat_PPR1_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS1_nf[u, 1] = MSPE_best_mars
  error_mat_XGB1_nf[u, 1] = MSPE_xgb
  
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim1_nf = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim1_nf[1, ] = colMeans(error_mat_lm1_nf, na.rm = T)
Final_Table_Sim1_nf[2, ] = colMeans(error_mat_lassoMin1_nf, na.rm = T)
Final_Table_Sim1_nf[3, ] = colMeans(error_mat_lasso1se1_nf, na.rm = T)
Final_Table_Sim1_nf[4, ] = colMeans(error_mat_RF1_nf, na.rm = T)
Final_Table_Sim1_nf[5, ] = colMeans(error_mat_GBM1_nf, na.rm = T)
Final_Table_Sim1_nf[6, ] = colMeans(error_mat_PPR1_nf, na.rm = T)
Final_Table_Sim1_nf[7, ] = colMeans(error_mat_MARS1_nf, na.rm = T) # this was bad for some reason so didn't include in results
Final_Table_Sim1_nf[8, ] = colMeans(error_mat_XGB1_nf, na.rm = T)

# Looking at results
Final_Table_Sim1_nf


#############################################################
# 2 - Exponential
#############################################################

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
error_mat_lm2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS2_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB2_nf = matrix(nrow = sim_num, ncol = 1)

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
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
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
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = TRUE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
  # Running FNN for bike (ONLY RUNNING THIS AGAIN TO MAKE SPLITS CONSISTENT)
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
  error_mat_lm2_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin2_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se2_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF2_nf[u, 1] = MSPE_rf
  error_mat_GBM2_nf[u, 1] = MSPE_gbm
  error_mat_PPR2_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS2_nf[u, 1] = MSPE_best_mars
  error_mat_XGB2_nf[u, 1] = MSPE_xgb
  
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  
}

# Initializing final table: average of errors
Final_Table_Sim2_nf = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim2_nf[1, ] = colMeans(error_mat_lm2_nf, na.rm = T)
Final_Table_Sim2_nf[2, ] = colMeans(error_mat_lassoMin2_nf, na.rm = T)
Final_Table_Sim2_nf[3, ] = colMeans(error_mat_lasso1se2_nf, na.rm = T)
Final_Table_Sim2_nf[4, ] = colMeans(error_mat_RF2_nf, na.rm = T)
Final_Table_Sim2_nf[5, ] = colMeans(error_mat_GBM2_nf, na.rm = T)
Final_Table_Sim2_nf[6, ] = colMeans(error_mat_PPR2_nf, na.rm = T)
Final_Table_Sim2_nf[7, ] = colMeans(error_mat_MARS2_nf, na.rm = T)
Final_Table_Sim2_nf[8, ] = colMeans(error_mat_XGB2_nf, na.rm = T)

# Looking at results
Final_Table_Sim2_nf


#############################################################
# 3 - Sigmoidal
#############################################################

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
error_mat_lm3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS3_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB3_nf = matrix(nrow = sim_num, ncol = 1)

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
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
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
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = TRUE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
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
  error_mat_lm3_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin3_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se3_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF3_nf[u, 1] = MSPE_rf
  error_mat_GBM3_nf[u, 1] = MSPE_gbm
  error_mat_PPR3_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS3_nf[u, 1] = MSPE_best_mars
  error_mat_XGB3_nf[u, 1] = MSPE_xgb
  
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
  
}

# Initializing final table: average of errors
Final_Table_Sim3_nf = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim3_nf[1, ] = colMeans(error_mat_lm3_nf, na.rm = T)
Final_Table_Sim3_nf[2, ] = colMeans(error_mat_lassoMin3_nf, na.rm = T)
Final_Table_Sim3_nf[3, ] = colMeans(error_mat_lasso1se3_nf, na.rm = T)
Final_Table_Sim3_nf[4, ] = colMeans(error_mat_RF3_nf, na.rm = T)
Final_Table_Sim3_nf[5, ] = colMeans(error_mat_GBM3_nf, na.rm = T)
Final_Table_Sim3_nf[6, ] = colMeans(error_mat_PPR3_nf, na.rm = T)
Final_Table_Sim3_nf[7, ] = colMeans(error_mat_MARS3_nf, na.rm = T)
Final_Table_Sim3_nf[8, ] = colMeans(error_mat_XGB3_nf, na.rm = T)

# Looking at results
Final_Table_Sim3_nf


#############################################################
# 4 - Log
#############################################################

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
error_mat_lm4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lassoMin4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_lasso1se4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_RF4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_GBM4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_PPR4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_MARS4_nf = matrix(nrow = sim_num, ncol = 1)
error_mat_XGB4_nf = matrix(nrow = sim_num, ncol = 1)

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
  
  # Setting up index
  ind = sample(1:300, 50)
  
  # Turning sim data into df
  sim_df = as.data.frame(sim_data)
  
  # appending
  sim_df2 = data.frame(resp = ystar, sim_df)
  
  # Factor data set
  train_f <- sim_df2[-ind,]
  test_f <- sim_df2[ind,]
  
  # Creating test and train data
  train_x <- sim_df2[-ind, -1]
  train_y <- sim_df2[-ind, 1]
  test_x <- sim_df2[ind, -1]
  test_y <- sim_df2[ind, 1]
  
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
  
  # Linear model
  mod_lm = lm(resp ~ ., data = train_f)
  pred_lm = predict(mod_lm, newdata = test_f)
  
  #################
  # LASSO
  #################
  
  y.1 <- train_y
  x.1 <- as.matrix(train_x)
  xs.1 <- scale(x.1)
  y.2 <- test_y
  x.2 <- as.matrix(test_x)
  xs.2 <- scale(x.2)
  
  # cv
  cv.lasso.1 <- cv.glmnet(y=y.1, x= x.1, family="gaussian")
  
  # Now predicting
  predict_lasso_min_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.min)
  predict_lasso_1se_mspe <- predict(cv.lasso.1, newx = x.2, s = cv.lasso.1$lambda.1se)
  
  #################
  # Random Forest
  #################
  
  # Creating grid to tune over
  tuning_par <- expand.grid(c(seq(1, 100, 20)), c(2, 4, 6, 8, 10))
  colnames(tuning_par) <- c("mtry", "nodesize")
  
  # Parallel applying
  plan(multiprocess, workers = 8)
  
  # Running through apply
  tuning_rf <- future_apply(tuning_par, 1, function(x){
    
    # Running Cross Validations
    rf_model <- randomForest(resp ~ ., data = train_f,
                             mtry = x[1],
                             nodesize = x[2])
    
    # Getting predictions
    sMSE = mean((predict(rf_model) - train_f$resp)^2)
    
    # Putting together
    df_returned <- data.frame(mtry = x[1], nodeisze = x[2], sMSE = sMSE)
    rownames(df_returned) <- NULL
    
    # Returning
    return(df_returned)
    
  })
  
  # Putting together results
  tuning_rf_results <- do.call(rbind, tuning_rf)
  
  # Saving Errors
  sMSE_rf_best <- tuning_rf_results[which.min(tuning_rf_results$sMSE), 3]
  
  # Getting MSPE
  final_rf <- randomForest(resp ~ ., data = train_f,
                           mtry = tuning_rf_results[which.min(tuning_rf_results$sMSE), 1],
                           nodesize = tuning_rf_results[which.min(tuning_rf_results$sMSE), 2])
  
  
  # Predicting for MSPE
  MSPE_rf <- mean((predict(final_rf, newdata = test_f) - test_f$resp)^2, na.rm = T)
  
  #################
  # GBM
  #################  
  
  # Building model
  gbm_model <- gbm(data = train_f, 
                   resp ~ ., 
                   distribution="gaussian", 
                   n.trees = 2000, 
                   interaction.depth = 7, 
                   shrinkage = 0.001, 
                   bag.fraction = 0.7,
                   n.minobsinnode = 11)
  
  # Tuned Model Prediction
  MSPE_gbm <- mean((test_f$resp - predict(gbm_model, 
                                          newdata = test_f, 
                                          n.trees=gbm_model$n.trees))^2, na.rm = T)
  
  #################
  # PPR
  #################
  
  ppr1 <- ppr(data = train_f, resp ~ ., 
              nterms = 1, optlevel = 3, sm.method = "gcvspline")
  
  ppr2 <- ppr(data = train_f, resp ~ ., 
              nterms = 2, optlevel = 3, sm.method = "gcvspline")
  
  ppr3 <- ppr(data = train_f, resp ~ ., 
              nterms = 3, optlevel = 3, sm.method = "gcvspline")
  
  ppr3_max6 <- ppr(data = train_f, resp ~ ., 
                   nterms = 3, optlevel = 3, sm.method = "gcvspline",
                   max.terms = 6)
  
  # Predicting
  MSPE_ppr_t1 <- mean((predict(ppr1, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t2 <- mean((predict(ppr2, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3 <- mean((predict(ppr3, newdata = test_f) - test_y)^2, na.rm = T)
  MSPE_ppr_t3_max6 <- mean((predict(ppr3_max6, newdata = test_f) - test_y)^2, na.rm = T)
  
  #################
  # MARS
  #################
  
  # Fitting 1 degree model
  mars_deg1_prune <- earth(resp ~ ., data = train_f, degree = 1,
                           pmethod = "backward")
  
  # Fitting 2 degree model
  mars_deg2_prune <- earth(resp ~ ., data = train_f, degree = 2,
                           pmethod = "backward")
  
  # Fitting 3 degree model
  mars_deg3_prune <- earth(resp ~ ., data = train_f, degree = 3,
                           pmethod = "backward")
  
  # Fitting 1 degree model, penalty = 5
  mars_deg1_prune5 <- earth(resp ~ ., data = train_f, degree = 1,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 2 degree model, penalty = 5
  mars_deg2_prune5 <- earth(resp ~ ., data = train_f, degree = 2,
                            pmethod = "backward", penalty = 5)
  
  # Fitting 3 degree model, penalty = 5
  mars_deg3_prune5 <- earth(resp ~ ., data = train_f, degree = 3,
                            pmethod = "backward", penalty = 5)
  
  # Getting MSPEs
  MSPE_deg1_prune <- mean((predict(mars_deg1_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune <- mean((predict(mars_deg2_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune <- mean((predict(mars_deg3_prune, test_f) - test_y)^2, na.rm = T)
  MSPE_deg1_prune5 <- mean((predict(mars_deg1_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg2_prune5 <- mean((predict(mars_deg2_prune5, test_f) - test_y)^2, na.rm = T)
  MSPE_deg3_prune5 <- mean((predict(mars_deg3_prune5, test_f) - test_y)^2, na.rm = T)
  
  # Finding minimum model
  best_model_num = which.max(c(mars_deg1_prune$grsq, mars_deg2_prune$grsq, 
                               mars_deg3_prune$grsq, mars_deg1_prune5$grsq, 
                               mars_deg2_prune5$grsq, mars_deg3_prune5$grsq))
  
  # Running best model
  MSPE_best_mars <- c(MSPE_deg1_prune, MSPE_deg2_prune, MSPE_deg3_prune,
                      MSPE_deg1_prune5, MSPE_deg2_prune5, MSPE_deg3_prune5)[best_model_num]
  
  #################
  ###### XGB ######
  #################  
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE,
    allowParallel = TRUE #
  )
  
  final_grid <- expand.grid(
    nrounds = 500,
    eta = c(0.05),
    max_depth = c(3),
    gamma = 0.5,
    colsample_bytree = 0.6,
    min_child_weight = 7,
    subsample = 0.7
  )
  
  xgb_model <- caret::train(
    x = train_x,
    y = train_y,
    trControl = train_control,
    tuneGrid = final_grid,
    method = "xgbTree",
    verbose = TRUE
  )
  
  # Predicting
  prediction_xgb <- predict(xgb_model, newdata = test_x)
  MSPE_xgb <- mean((test_y - prediction_xgb)^2, na.rm = T)
  
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
  
  # MSPE Results
  error_mat_lm4_nf[u, 1] = mean((pred_lm - test_y)^2, na.rm = T)
  error_mat_lassoMin4_nf[u, 1] = mean((y.2 - predict_lasso_min_mspe)^2, na.rm = T)
  error_mat_lasso1se4_nf[u, 1] = mean((y.2 - predict_lasso_1se_mspe)^2, na.rm = T)
  error_mat_RF4_nf[u, 1] = MSPE_rf
  error_mat_GBM4_nf[u, 1] = MSPE_gbm
  error_mat_PPR4_nf[u, 1] = min(MSPE_ppr_t1, MSPE_ppr_t2, MSPE_ppr_t3, MSPE_ppr_t3_max6)
  error_mat_MARS4_nf[u, 1] = MSPE_best_mars
  error_mat_XGB4_nf[u, 1] = MSPE_xgb
  
  
  # Printing iteration number
  print(paste0("Done Iteration: ", u))
  
  # Clearing backend
  K <- backend()
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Sim4_nf = matrix(nrow = 8, ncol = 1)

# Collecting errors
Final_Table_Sim4_nf[1, ] = colMeans(error_mat_lm4_nf, na.rm = T)
Final_Table_Sim4_nf[2, ] = colMeans(error_mat_lassoMin4_nf, na.rm = T)
Final_Table_Sim4_nf[3, ] = colMeans(error_mat_lasso1se4_nf, na.rm = T)
Final_Table_Sim4_nf[4, ] = colMeans(error_mat_RF4_nf, na.rm = T)
Final_Table_Sim4_nf[5, ] = colMeans(error_mat_GBM4_nf, na.rm = T)
Final_Table_Sim4_nf[6, ] = colMeans(error_mat_PPR4_nf, na.rm = T)
Final_Table_Sim4_nf[7, ] = colMeans(error_mat_MARS4_nf, na.rm = T)
Final_Table_Sim4_nf[8, ] = colMeans(error_mat_XGB4_nf, na.rm = T)

# Looking at results
Final_Table_Sim4_nf


#####################################################################################
#####################################################################################
# Comparing Using Boxplots - ONLY RUN THIS PART IF YOU HAVE RAN
# FNN - Simulation Predictions -Final.R. That is, in order to reproduce
# the results, you must first run "FNN - Simulation Predictions - Final.r",
# then (next) you need to run the code above in this file, and then finally, 
# you can run the code below. This will return the prediction boxplots in the paper
#####################################################################################
#####################################################################################

# Making matrices for each simulation
sim1_mat = cbind(error_mat_lm1_nf, 
                 error_mat_lassoMin1_nf,
                 error_mat_lasso1se1_nf,
                 error_mat_RF1_nf,
                 error_mat_GBM1_nf,
                 error_mat_PPR1_nf,
                 error_mat_XGB1_nf,
                 error_mat_lm_1,
                 error_mat_pc1_1,
                 error_mat_pc2_1,
                 error_mat_pc3_1,
                 error_mat_pls1_1,
                 error_mat_pls2_1,
                 error_mat_fnn_1)

sim2_mat = cbind(error_mat_lm2_nf, 
                 error_mat_lassoMin2_nf,
                 error_mat_lasso1se2_nf,
                 error_mat_RF2_nf,
                 error_mat_GBM2_nf,
                 error_mat_PPR2_nf,
                 error_mat_XGB2_nf,
                 error_mat_lm_2,
                 error_mat_pc1_2,
                 error_mat_pc2_2,
                 error_mat_pc3_2,
                 error_mat_pls1_2,
                 error_mat_pls2_2,
                 error_mat_fnn_2)

sim3_mat = cbind(error_mat_lm3_nf, 
                 error_mat_lassoMin3_nf,
                 error_mat_lasso1se3_nf,
                 error_mat_RF3_nf,
                 error_mat_GBM3_nf,
                 error_mat_PPR3_nf,
                 error_mat_XGB3_nf,
                 error_mat_lm_3,
                 error_mat_pc1_3,
                 error_mat_pc2_3,
                 error_mat_pc3_3,
                 error_mat_pls1_3,
                 error_mat_pls2_3,
                 error_mat_fnn_3)

sim4_mat = cbind(error_mat_lm4_nf, 
                 error_mat_lassoMin4_nf,
                 error_mat_lasso1se4_nf,
                 error_mat_RF4_nf,
                 error_mat_GBM4_nf,
                 error_mat_PPR4_nf,
                 error_mat_XGB4_nf,
                 error_mat_lm_4,
                 error_mat_pc1_4,
                 error_mat_pc2_4,
                 error_mat_pc3_4,
                 error_mat_pls1_4,
                 error_mat_pls2_4,
                 error_mat_fnn_4)

# Names
names_list = c("MLR", "LASSO_Min", "LASOO_1se", "RF", "GBM", "PPR", "XGB", 
               "FLM", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "FNN")

# Changing
colnames(sim1_mat) = names_list
colnames(sim2_mat) = names_list
colnames(sim3_mat) = names_list
colnames(sim4_mat) = names_list

# Saving matrices
write.table(sim1_mat, file="sim1Pred.csv", row.names = F)
write.table(sim2_mat, file="sim2Pred.csv", row.names = F)
write.table(sim3_mat, file="sim3Pred.csv", row.names = F)
write.table(sim4_mat, file="sim4Pred.csv", row.names = F)

# Creating boxplots

#### Sim 1

# Saving sqrt
sqrt_MSPE <- data.frame(sqrt(sim1_mat))

# Creating boxplots
plot1 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
    geom_boxplot(fill='#A4A4A4', color="darkred") + 
    theme_bw() + 
    xlab("Model\nSimulation: 1") +
    ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 1)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot1

#### Sim 2

# Saving sqrt
sqrt_MSPE <- data.frame(sim2_mat)

# Creating boxplots
plot2 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
    geom_boxplot(fill='#A4A4A4', color="darkred") + 
    theme_bw() + 
    xlab("Model\nSimulation: 2") +
    ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot2

#### Sim 3

# Saving sqrt
sqrt_MSPE <- data.frame(sqrt(sim3_mat))

# Creating boxplots
plot3 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
    geom_boxplot(fill='#A4A4A4', color="darkred") + 
    theme_bw() + 
    xlab("Model\nSimulation: 3") +
    ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 0.2)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot3

#### Sim 4

# Saving sqrt
sqrt_MSPE <- data.frame(sqrt(sim4_mat))

# Creating boxplots
plot4 <- ggplot(stack(sqrt_MSPE), aes(x = ind, y = values)) +
    geom_boxplot(fill='#A4A4A4', color="darkred") + 
    theme_bw() + 
    xlab("Model\nSimulation: 4") +
    ylab("MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 3)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold"))

plot4


###################### RELATIVE PLOTS ##########################

# Getting minimums
mspe_div1_mins = apply(sim1_mat, 1, function(x){
  return(min(x))
})

mspe_div2_mins = apply(sim2_mat, 1, function(x){
  return(min(x))
})

mspe_div3_mins = apply(sim3_mat, 1, function(x){
  return(min(x))
})

mspe_div4_mins = apply(sim4_mat, 1, function(x){
  return(min(x))
})

# Initializing
mspe_div1 = matrix(nrow = nrow(sim1_mat), ncol = ncol(sim1_mat))
mspe_div2 = matrix(nrow = nrow(sim2_mat), ncol = ncol(sim2_mat))
mspe_div3 = matrix(nrow = nrow(sim3_mat), ncol = ncol(sim3_mat))
mspe_div4 = matrix(nrow = nrow(sim4_mat), ncol = ncol(sim4_mat))

for (i in 1:sim_num) {
  mspe_div1[i, ] = sim1_mat[i,]/mspe_div1_mins[i]
  mspe_div2[i, ] = sim2_mat[i,]/mspe_div2_mins[i]
  mspe_div3[i, ] = sim3_mat[i,]/mspe_div3_mins[i]
  mspe_div4[i, ] = sim4_mat[i,]/mspe_div4_mins[i]
  
}

# names
colnames(mspe_div1) = names_list
colnames(mspe_div2) = names_list
colnames(mspe_div3) = names_list
colnames(mspe_div4) = names_list

# Creating relative boxplots

# turning into df
df_MSPE <- data.frame(mspe_div1)

# Creating boxplots
plot1_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 1\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme(axis.text.x = element_blank())

plot1_rel

# turning into df
df_MSPE <- data.frame(mspe_div2)

# Creating boxplots
plot2_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 2\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme(axis.text.x = element_blank())

plot2_rel

# turning into df
df_MSPE <- data.frame(mspe_div3)

# Creating boxplots
plot3_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 3\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme(axis.text.x = element_blank())

plot3_rel

# turning into df
df_MSPE <- data.frame(mspe_div4)

# Creating boxplots
plot4_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 4\nRelative MSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 5)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 1, linetype = "dashed")

plot4_rel

# Some libraries
library(FinCal)
library(reshape2)
library(grid)

# Saving plots
grid.draw(rbind(ggplotGrob(plot1_rel), ggplotGrob(plot2_rel), ggplotGrob(plot3_rel), ggplotGrob(plot4_rel), size = "last"))


# Check 1