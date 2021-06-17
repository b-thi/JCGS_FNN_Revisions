##############################
#                            #
# Weather Data Set - Final   #
#                            #
##############################

##############################
# Data Information:
#
# Weather Data Set
# Observations: 35
# Continuum Points: 365
# Domain: [1, 365]
# Basis Functions used for Functional Observations: 
# Range of Response: [a, b]
# Basis Functions used for Functional Weights: 
# Folds Used: LOOCV
# Parameter Count in FNN:
# Parameter Count in CNN:
# Parameter Count in NN:
##############################

# Libraries
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
daily = readRDS("Data/daily.RDS")

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
num_folds = 5

# Creating folds
fold_ind = createFolds(total_prec, k = num_folds)

# Initializing matrices for results
error_mat_lm = matrix(nrow = num_folds, ncol = 1)
error_mat_pc1 = matrix(nrow = num_folds, ncol = 1)
error_mat_pc2 = matrix(nrow = num_folds, ncol = 1)
error_mat_pc3 = matrix(nrow = num_folds, ncol = 1)
error_mat_pls1 = matrix(nrow = num_folds, ncol = 1)
error_mat_pls2 = matrix(nrow = num_folds, ncol = 1)
error_mat_np = matrix(nrow = num_folds, ncol = 1)
error_mat_cnn = matrix(nrow = num_folds, ncol = 1)
error_mat_nn = matrix(nrow = num_folds, ncol = 1)
error_mat_fnn = matrix(nrow = num_folds, ncol = 1)

# Doing pre-processing of neural networks
if(dim(temp_data)[3] > 1){
  # Now, let's pre-process
  pre_dat = FNN_First_Layer(func_cov = temp_data,
                            basis_choice = c("fourier", "fourier", "fourier"),
                            num_basis = c(5, 7, 9),
                            domain_range = list(c(min(timepts), max(timepts)), 
                                                c(min(timepts), max(timepts)), 
                                                c(min(timepts), max(timepts))),
                            covariate_scaling = T,
                            raw_data = F)
  
} else {
  
  # Now, let's pre-process
  pre_dat = FNN_First_Layer(func_cov = temp_data,
                            basis_choice = c("bspline"),
                            num_basis = c(19),
                            domain_range = list(c(min(timepts), max(timepts))),
                            covariate_scaling = T,
                            raw_data = F)
}

fd_test = Data2fd(timepts, t(pre_dat$data), tempbasis65)

# Functional weights
func_weights = matrix(nrow = num_folds, ncol = 5)
flm_weights = list()
nn_training_plot <- list()
cnn_training_plot <- list()
fnn_training_plot <- list()

# Testing
# i = 1
# u = 1

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
  # weather_data_train <- array(dim = c(65, ncol(temp_data) - length(fold_ind[[i]]), 1))
  # weather_data_test <- array(dim = c(65, length(fold_ind[[i]]), 1))
  # weather_data_train[,,1] = temp_data[, -fold_ind[[i]], ]
  # weather_data_test[,,1] = temp_data[, fold_ind[[i]], ]
  
  # Setting up for FNN
  pre_train = pre_dat$data[-fold_ind[[i]], ]
  pre_test = pre_dat$data[fold_ind[[i]], ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional Linear Model (Basis)
  l=2^(-4:10)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = predict(func_basis[[1]], test_x)
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 6)
  pred_pc = predict(func_pc$fregre.pc, test_x)
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 6, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:6, lambda=TRUE, P=1)
  pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:6)
  pred_pls = predict(func_pls, test_x)
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 6, lambda=0:5, P=c(0,0,1))
  pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = predict(func_np, test_x)
  
  ########################################
  # Neural Network Tuning Setup          #
  ########################################
  
  # Initializing
  min_error_nn = 99999
  min_error_cnn = 99999
  min_error_fnn = 99999
  
  # Setting up MV data
  MV_train = as.data.frame(t(daily$tempav)[-fold_ind[[i]],])
  MV_test = as.data.frame(t(daily$tempav)[fold_ind[[i]],])
  
  # Random Split
  train_split = sample(1:nrow(MV_train), floor(0.75*nrow(MV_train)))
  
  # Learn rates grid
  num_initalizations = 1
  
  ########################################
  # Running Convolutional Neural Network #
  ########################################
  
  # Setting seeds
  # set.seed(i)
  # use_session_with_seed(
  #   i,
  #   disable_gpu = F,
  #   disable_parallel_cpu = F,
  #   quiet = T
  # )
  
  # Setting up CNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_cnn <- keras_model_sequential()
    model_cnn %>% 
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu", 
                    input_shape = c(ncol(MV_train[train_split,]), 1)) %>% 
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      layer_flatten() %>% 
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'sigmoid')
    
    # Setting parameters for NN model
    model_cnn %>% compile(
      optimizer = optimizer_adam(lr = 0.00005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(MV_train[train_split,]), ncol(MV_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(MV_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(MV_train[-train_split,]), ncol(MV_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(MV_train[-train_split,])
    
    # Training CNN model
    history_cnn <- model_cnn %>% fit(reshaped_data_tensor_train, 
                                     train_y[train_split], 
                                     epochs = 5000,  
                                     validation_split = 0.2,
                                     callbacks = list(early_stop),
                                     verbose = 0)
    
    # Predictions
    test_predictions <- model_cnn %>% predict(reshaped_data_tensor_test)
    
    # Plotting
    error_cnn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_cnn_train < min_error_cnn){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(MV_test), ncol(MV_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(MV_test)
      
      # Predictions
      pred_cnn <- model_cnn %>% predict(reshaped_data_tensor_test_final)
      
      # Saving training plots
      cnn_training_plot[[i]] = as.data.frame(history_cnn)
      
      # Error
      error_cnn = mean((c(pred_cnn) - test_y)^2, na.rm = T)
      
      # New Min Error
      min_error_cnn = error_cnn_train
      
    }
    
  }
  
  ########################################
  # Running Conventional Neural Network  #
  ########################################
  
  # Setting seeds
  # set.seed(i)
  # use_session_with_seed(
  #   i,
  #   disable_gpu = F,
  #   disable_parallel_cpu = F,
  #   quiet = T
  # )
  
  # Setting up NN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_nn <- keras_model_sequential()
    model_nn %>% 
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'sigmoid')
    
    # Setting parameters for NN model
    model_nn %>% compile(
      optimizer = optimizer_adam(lr = 0.00005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
    
    # Training FNN model
    history_nn <- model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                                   train_y[train_split], 
                                   epochs = 5000,  
                                   validation_split = 0.2,
                                   callbacks = list(early_stop),
                                   verbose = 0)
    
    # Predictions
    test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
    
    # Plotting
    error_nn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    
    # Checking error
    if(error_nn_train < min_error_nn){
      
      # Predictions
      pred_nn <- model_nn %>% predict(as.matrix(MV_test))
      
      # Error
      error_nn = mean((c(pred_nn) - test_y)^2, na.rm = T)
      
      # Saving training plots
      nn_training_plot[[i]] = as.data.frame(history_nn)
      
      # New Min Error
      min_error_nn = error_nn_train
      
    }
    
  }
  
  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Setting seeds
  # set.seed(i)
  # use_session_with_seed(
  #   i,
  #   disable_gpu = F,
  #   disable_parallel_cpu = F,
  #   quiet = T
  # )
  
  # Setting up FNN model
  for(u in 1:num_initalizations){
    
    # setting up model
    model_fnn <- keras_model_sequential()
    model_fnn %>% 
      # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu",
      #               input_shape = c(ncol(pre_train[train_split,]), 1)) %>%
      # layer_max_pooling_1d(pool_size = 2) %>%
      # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
      # layer_flatten() %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 64, activation = 'relu') %>%
      layer_dense(units = 1, activation = 'sigmoid')
    
    # Setting parameters for FNN model
    model_fnn %>% compile(
      optimizer = optimizer_adam(lr = 0.00005), 
      loss = 'mse',
      metrics = c('mean_squared_error')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
    
    # Setting up data
    reshaped_data_tensor_train = array(dim = c(nrow(pre_train[train_split,]), ncol(pre_train[train_split,]), 1))
    reshaped_data_tensor_train[, , 1] = as.matrix(pre_train[train_split,])
    reshaped_data_tensor_test = array(dim = c(nrow(pre_train[-train_split,]), ncol(pre_train[-train_split,]), 1))
    reshaped_data_tensor_test[, , 1] = as.matrix(pre_train[-train_split,])
    
    # Training FNN model
    # history_fnn <- model_fnn %>% fit(reshaped_data_tensor_train,
    #                  train_y[train_split],
    #                  epochs = 5000,
    #                  validation_split = 0.2,
    #                  callbacks = list(early_stop),
    #                  verbose = 0)
    
    # Training FNN model
    history_fnn = model_fnn %>% fit(pre_train[train_split,],
                      train_y[train_split],
                      epochs = 5000,
                      validation_split = 0.2,
                      callbacks = list(early_stop),
                      verbose = 1)
    
    # Predictions
    test_predictions <- model_fnn %>% predict(pre_train[-train_split,])
    # test_predictions <- model_fnn %>% predict(reshaped_data_tensor_test)
    
    # Storing
    # error_fnn_train = mean((c(test_predictions) - train_y[-train_split])^2)
    error_fnn_train = mean((test_predictions - train_y[-train_split])^2, na.rm = T)
    
    # Checking error
    if(error_fnn_train < min_error_fnn){
      
      # Setting up test data
      reshaped_data_tensor_test_final = array(dim = c(nrow(pre_test), ncol(pre_test), 1))
      reshaped_data_tensor_test_final[, , 1] = as.matrix(pre_test)
      
      # Predictions
      pred_fnn <- model_fnn %>% predict(pre_test)
      # pred_fnn <- model_fnn %>% predict(reshaped_data_tensor_test_final)
      
      # Error
      error_fnn = mean((c(pred_fnn) - test_y)^2, na.rm = T)
      
      # Saving training plots
      fnn_training_plot[[i]] = as.data.frame(history_fnn)
      
      # New Min Error
      min_error_fnn = error_fnn_train
      
    }
    
  }

  
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
  error_mat_cnn[i, 1] = mean((pred_cnn - test_y)^2, na.rm = T)
  error_mat_nn[i, 1] = mean((pred_nn - test_y)^2, na.rm = T)
  error_mat_fnn[i, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  # error_mat_lm[i, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc1[i, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc2[i, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc3[i, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls1[i, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls2[i, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_np[i, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_cnn[i, 2] = 1 - sum((pred_cnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_nn[i, 2] = 1 - sum((pred_nn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_fnn[i, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
 
  # Printing iteration number
  print(paste0("Done Iteration: ", i))
  
  # Clearing session
  K$clear_session()
  
}

# Initializing final table: average of errors
Final_Table_Weather = matrix(nrow = 10, ncol = 2)

# Collecting errors
Final_Table_Weather[1, 1] = mean(error_mat_lm[,1], na.rm = T)
Final_Table_Weather[2, 1] = mean(error_mat_np[,1], na.rm = T)
Final_Table_Weather[3, 1] = mean(error_mat_pc1[,1], na.rm = T)
Final_Table_Weather[4, 1] = mean(error_mat_pc2[,1], na.rm = T)
Final_Table_Weather[5, 1] = mean(error_mat_pc3[,1], na.rm = T)
Final_Table_Weather[6, 1] = mean(error_mat_pls1[,1], na.rm = T)
Final_Table_Weather[7, 1] = mean(error_mat_pls2[,1], na.rm = T)
Final_Table_Weather[8, 1] = mean(error_mat_cnn[,1], na.rm = T)
Final_Table_Weather[9, 1] = mean(error_mat_nn[,1], na.rm = T)
Final_Table_Weather[10, 1] = mean(error_mat_fnn[,1], na.rm = T)

# R_Squared value
# Final_Table_Weather[1, 2] = mean(error_mat_lm[,2], na.rm = T)
# Final_Table_Weather[2, 2] = mean(error_mat_np[,2], na.rm = T)
# Final_Table_Weather[3, 2] = mean(error_mat_pc1[,2], na.rm = T)
# Final_Table_Weather[4, 2] = mean(error_mat_pc2[,2], na.rm = T)
# Final_Table_Weather[5, 2] = mean(error_mat_pc3[,2], na.rm = T)
# Final_Table_Weather[6, 2] = mean(error_mat_pls1[,2], na.rm = T)
# Final_Table_Weather[7, 2] = mean(error_mat_pls2[,2], na.rm = T)
# Final_Table_Weather[8, 2] = mean(error_mat_cnn[,2], na.rm = T)
# Final_Table_Weather[9, 2] = mean(error_mat_nn[,2], na.rm = T)
# Final_Table_Weather[10, 2] = mean(error_mat_fnn[,2], na.rm = T)

# Standard error
Final_Table_Weather[1, 2] = sd(c(error_mat_lm[,1]), na.rm = T)/sqrt(num_folds)
Final_Table_Weather[2, 2] = sd(error_mat_np[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[3, 2] = sd(error_mat_pc1[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[4, 2] = sd(error_mat_pc2[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[5, 2] = sd(error_mat_pc3[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[6, 2] = sd(error_mat_pls1[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[7, 2] = sd(error_mat_pls2[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[8, 2] = sd(error_mat_cnn[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[9, 2] = sd(error_mat_nn[,1], na.rm = T)/sqrt(num_folds)
Final_Table_Weather[10, 2] = sd(error_mat_fnn[,1], na.rm = T)/sqrt(num_folds)

# Looking at results
colnames(Final_Table_Weather) <- c("CV_MSPE", "SE")
rownames(Final_Table_Weather) <- c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN")
Final_Table_Weather

# Training plots saving

# Initializing plots
training_plots_weather = list()

# Looping
for (i in 1:num_folds) {
  
  # count
  a = 3*(i - 1)
  
  # Saving relevant
  current_cnn = cnn_training_plot[[i]]
  current_nn = nn_training_plot[[i]]
  current_fnn = fnn_training_plot[[i]]
  
  # Filtering
  current_cnn = current_cnn %>% dplyr::filter(metric == "loss" & data == "validation")
  current_nn = current_nn %>% dplyr::filter(metric == "loss" & data == "validation")
  current_fnn = current_fnn %>% dplyr::filter(metric == "loss" & data == "validation")
  
  # Creating plots
  cnn_plot = current_cnn %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5,  color='red') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Convolutional Neural Network; Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=12, face = "bold"),
          axis.title=element_text(size=12,face="bold"))
  
  nn_plot = current_nn %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5, color='green') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Conventional Neural Network; Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=12, face = "bold"),
          axis.title=element_text(size=12,face="bold"))
  
  fnn_plot = current_fnn %>% 
    ggplot(aes(x = epoch, y = value)) +
    geom_line(size = 1.5, color='blue') + 
    theme_bw() +
    xlab("Epoch") +
    ylab("Validation Loss") +
    ggtitle(paste("Functional Neural Network; Fold: ", i)) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text=element_text(size=12, face = "bold"),
          axis.title=element_text(size=12,face="bold"))
  
  
  # Storing
  training_plots_weather[[a + 1]] = cnn_plot
  training_plots_weather[[a + 2]] = nn_plot
  training_plots_weather[[a + 3]] = fnn_plot
  
  
}

# Final Plot
n_plots <- length(training_plots_weather)
nCol <- floor(sqrt(n_plots))
do.call("grid.arrange", c(training_plots_weather, ncol = nCol))

# Functional Weight Plot

# Setting up data set
beta_coef_fnn <- data.frame(time = seq(1, 24, 0.1), 
                            beta_evals = beta_fnn_weather(seq(1, 24, 0.1), colMeans(func_weights)))

# Plot
beta_coef_fnn %>% 
  ggplot(aes(x = time, y = beta_evals, color='blue')) +
  geom_line(size = 1.5) + 
  theme_bw() +
  xlab("Time") +
  ylab("beta(t)") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text=element_text(size=14, face = "bold"),
        axis.title=element_text(size=14,face="bold")) +
  scale_colour_manual(name = 'Model: ', 
                      values =c('blue'='blue'), 
                      labels = c('Functional Neural Network')) +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid", 
                                         colour ="darkblue"),
        legend.position = "bottom",
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

# Running paired t-tests

# Creating data frame
t_test_df = cbind(error_mat_lm[, 1],
                  error_mat_np[, 1],
                  error_mat_pc1[, 1],
                  error_mat_pc2[, 1],
                  error_mat_pc3[, 1],
                  error_mat_pls1[, 1],
                  error_mat_pls2[, 1],
                  error_mat_cnn[, 1],
                  error_mat_nn[, 1],
                  error_mat_fnn[, 1])

# Initializing
p_value_df = matrix(nrow = ncol(t_test_df), ncol = 4)
rownames(p_value_df) = c("FLM", "FNP", "FPC", "FPC_Deriv", "FPC_Ridge", "FPLS", "FPLS_Deriv", "CNN", "NN", "FNN")
colnames(p_value_df) = c("P Value", "T Value", "Lower Bound", "Upper Upper Bound")

# Getting p-values
for(i in 1:ncol(t_test_df)) {
  
  # Selecting data sets
  FNN_ttest = t_test_df[, 10]
  Other_ttest = t_test_df[, i]
  
  # Calculating difference
  d = FNN_ttest - Other_ttest
  
  # Mean difference
  mean_d = mean(d)
  
  # SE
  se_d = sd(d)/sqrt(length(FNN_ttest))
  
  # T value
  T_value = mean_d/se_d
  
  # df
  df_val = length(FNN_ttest) - 1
  
  # p-value
  p_value = pt(abs(T_value), df_val, lower.tail = F)
  
  # Storing
  p_value_df[i, 1] = p_value
  p_value_df[i, 2] = T_value
  p_value_df[i, 3] = mean_d - T_value*se_d
  p_value_df[i, 4] = mean_d + T_value*se_d
}