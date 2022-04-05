# Commented Out Code

###### Bike

# Doing pre-processing of neural networks
# if(dim(bike_data)[3] > 1){
#   # Now, let's pre-process
#   pre_dat = FNN_First_Layer(func_cov = bike_data,
#                            basis_choice = c("fourier", "fourier", "fourier"),
#                            num_basis = c(5, 7, 9),
#                            domain_range = list(c(min(timepts), max(timepts)), 
#                                                c(min(timepts), max(timepts)), 
#                                                c(min(timepts), max(timepts))),
#                            covariate_scaling = T,
#                            raw_data = F)
#   
# } else {
#   
#   # Now, let's pre-process
#   pre_dat = FNN_First_Layer(func_cov = bike_data,
#                            basis_choice = c("fourier"),
#                            num_basis = c(17),
#                            domain_range = list(c(min(timepts), max(timepts))),
#                            covariate_scaling = T,
#                            raw_data = F)
# }

# # Setting up FNN model
# for(u in 1:num_initalizations){
#   
#   # setting up model
#   model_fnn <- keras_model_sequential()
#   model_fnn %>% 
#     # layer_dense(units = 1, activation = 'sigmoid') %>%
#     # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu",
#     #               input_shape = c(ncol(pre_train[train_split,]), 1)) %>%
#     # layer_max_pooling_1d(pool_size = 2) %>%
#     # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
#     # layer_flatten() %>%
#     layer_dense(units = 64, activation = 'sigmoid') %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 1, activation = 'linear')
#   
#   # Setting parameters for FNN model
#   model_fnn %>% compile(
#     optimizer = optimizer_adam(lr = 0.00005), 
#     loss = 'mse',
#     metrics = c('mean_squared_error')
#   )
#   
#   # Early stopping
#   early_stop <- callback_early_stopping(monitor = "val_loss", patience = 100)
#   
#   # Setting up data
#   reshaped_data_tensor_train = array(dim = c(nrow(pre_train[train_split,]), ncol(pre_train[train_split,]), 1))
#   reshaped_data_tensor_train[, , 1] = as.matrix(pre_train[train_split,])
#   reshaped_data_tensor_test = array(dim = c(nrow(pre_train[-train_split,]), ncol(pre_train[-train_split,]), 1))
#   reshaped_data_tensor_test[, , 1] = as.matrix(pre_train[-train_split,])
#   
#   # Training FNN model
#   # history_fnn <- model_fnn %>% fit(reshaped_data_tensor_train,
#   #                  train_y[train_split],
#   #                  epochs = 5000,
#   #                  validation_split = 0.2,
#   #                  callbacks = list(early_stop),
#   #                  verbose = 0)
#   
#   # Training FNN model
#   history_fnn = model_fnn %>% fit(pre_train[train_split,],
#                    train_y[train_split],
#                    epochs = 5000,
#                    validation_split = 0.2,
#                    callbacks = list(early_stop),
#                    verbose = 0)
#   
#   # Predictions
#   test_predictions <- model_fnn %>% predict(pre_train[-train_split,])
#   # test_predictions <- model_fnn %>% predict(reshaped_data_tensor_test)
#   
#   # Storing
#   # error_fnn_train = mean((c(test_predictions) - train_y[-train_split])^2)
#   error_fnn_train = mean((test_predictions - train_y[-train_split])^2, na.rm = T)
#   
#   # Checking error
#   if(error_fnn_train < min_error_fnn){
#     
#     # Setting up test data
#     # reshaped_data_tensor_test_final = array(dim = c(nrow(pre_test), ncol(pre_test), 1))
#     # reshaped_data_tensor_test_final[, , 1] = as.matrix(pre_test)
#     
#     # Predictions
#     pred_fnn <- model_fnn %>% predict(pre_test)
#     # pred_fnn <- model_fnn %>% predict(reshaped_data_tensor_test_final)
#     
#     # Error
#     error_fnn = mean((c(pred_fnn) - test_y)^2, na.rm = T)
#     
#     # Saving training plots
#     fnn_training_plot[[i]] = as.data.frame(history_fnn)
#     
#     # New Min Error
#     min_error_fnn = error_fnn_train
#     
#   }
#   
# }

# for(u in 1:num_initalizations){
#   
#   # Getting subset data
#   bike_data_train_tune <- array(dim = c(31, length(train_split), 1))
#   bike_data_test_tune <- array(dim = c(31, nrow(MV_train) - length(train_split), 1))
#   bike_data_train_tune[,,1] = bike_data_train[, train_split, ]
#   bike_data_test_tune[,,1] = bike_data_train[, -train_split, ]
#   
#   # Running FNN for bike
#   fnn_example = FNN(resp = train_y[train_split], 
#                     func_cov = bike_data_train_tune, 
#                     scalar_cov = NULL,
#                     basis_choice = c("fourier"), 
#                     num_basis = c(5),
#                     hidden_layers = 2,
#                     neurons_per_layer = c(32, 32),
#                     activations_in_layers = c("sigmoid", "relu"),
#                     domain_range = list(c(1, 24)),
#                     epochs = 500,
#                     output_size = 1,
#                     loss_choice = "mse",
#                     metric_choice = list("mean_squared_error"),
#                     val_split = 0.15,
#                     learn_rate = 0.002,
#                     patience_param = 15,
#                     early_stop = T,
#                     print_info = F)
#   
#   # Predicting using FNN for bike
#   pred_fnn = FNN_Predict(fnn_example,
#                          bike_data_test_tune, 
#                          scalar_cov = NULL,
#                          basis_choice = c("fourier"), 
#                          num_basis = c(5),
#                          domain_range = list(c(1, 24)))
#   
#   # Checking error
#   error_fnn_train = mean((pred_fnn - train_y[-train_split])^2, na.rm = T)
#   
#   # Checking error
#   if(error_fnn_train < min_error_fnn){
#     
#     # Predictions
#     pred_fnn <- FNN_Predict(fnn_example,
#                             bike_data_test, 
#                             scalar_cov = NULL,
#                             basis_choice = c("fourier"), 
#                             num_basis = c(5),
#                             domain_range = list(c(1, 24)))
#     
#     # Error
#     error_fnn = mean((c(pred_fnn) - test_y)^2, na.rm = T)
#     
#     # New Min Error
#     min_error_fnn = error_fnn_train
#     
#   }
#   
#   
# }

######## Tecator

### Creating Actual v. Predicted Plot
# actual_v_predicted = ggplot(data = data.frame(pred_fnn = pred_tec, actual = tecResp_test), aes(x = pred_fnn, y = actual)) +
#   theme_bw() +
#   geom_smooth(aes(color = "blue"), color = "blue", se = F) +
#   geom_smooth(data = data.frame(pred_nn = pred_nn[,1], actual = tecResp_test), aes(x = pred_nn, y = actual, color = "green"), color = "green", se = F) +
#   geom_smooth(data = data.frame(pred_cnn = pred_cnn[,1], actual = tecResp_test), aes(x = pred_cnn, y = actual, color = "red"), color = "red", se = F) +
#   ggtitle(paste("Actual v. Predicted Plots")) +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(axis.text=element_text(size=12, face = "bold"),
#         axis.title=element_text(size=12,face="bold")) +
#   xlab("Predicted") +
#   ylab("Actual")

####### Weather

# Doing pre-processing of neural networks
# if(dim(temp_data)[3] > 1){
#   # Now, let's pre-process
#   pre_dat = FNN_First_Layer(func_cov = temp_data,
#                             basis_choice = c("fourier", "fourier", "fourier"),
#                             num_basis = c(5, 7, 9),
#                             domain_range = list(c(min(timepts), max(timepts)), 
#                                                 c(min(timepts), max(timepts)), 
#                                                 c(min(timepts), max(timepts))),
#                             covariate_scaling = T,
#                             raw_data = F)
#   
# } else {
#   
#   # Now, let's pre-process
#   pre_dat = FNN_First_Layer(func_cov = temp_data,
#                             basis_choice = c("bspline"),
#                             num_basis = c(19),
#                             domain_range = list(c(min(timepts), max(timepts))),
#                             covariate_scaling = T,
#                             raw_data = F)
# }

# # Setting up FNN model
# for(u in 1:num_initalizations){
#   
#   # setting up model
#   model_fnn <- keras_model_sequential()
#   model_fnn %>% 
#     # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu",
#     #               input_shape = c(ncol(pre_train[train_split,]), 1)) %>%
#     # layer_max_pooling_1d(pool_size = 2) %>%
#     # layer_conv_1d(filters = 64, kernel_size = 2, activation = "relu") %>%
#     # layer_flatten() %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 64, activation = 'relu') %>%
#     layer_dense(units = 1, activation = 'sigmoid')
#   
#   # Setting parameters for FNN model
#   model_fnn %>% compile(
#     optimizer = optimizer_adam(lr = 0.00005), 
#     loss = 'mse',
#     metrics = c('mean_squared_error')
#   )
#   
#   # Early stopping
#   early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)
#   
#   # Setting up data
#   reshaped_data_tensor_train = array(dim = c(nrow(pre_train[train_split,]), ncol(pre_train[train_split,]), 1))
#   reshaped_data_tensor_train[, , 1] = as.matrix(pre_train[train_split,])
#   reshaped_data_tensor_test = array(dim = c(nrow(pre_train[-train_split,]), ncol(pre_train[-train_split,]), 1))
#   reshaped_data_tensor_test[, , 1] = as.matrix(pre_train[-train_split,])
#   
#   # Training FNN model
#   # history_fnn <- model_fnn %>% fit(reshaped_data_tensor_train,
#   #                  train_y[train_split],
#   #                  epochs = 5000,
#   #                  validation_split = 0.2,
#   #                  callbacks = list(early_stop),
#   #                  verbose = 0)
#   
#   # Training FNN model
#   history_fnn = model_fnn %>% fit(pre_train[train_split,],
#                     train_y[train_split],
#                     epochs = 5000,
#                     validation_split = 0.2,
#                     callbacks = list(early_stop),
#                     verbose = 1)
#   
#   # Predictions
#   test_predictions <- model_fnn %>% predict(pre_train[-train_split,])
#   # test_predictions <- model_fnn %>% predict(reshaped_data_tensor_test)
#   
#   # Storing
#   # error_fnn_train = mean((c(test_predictions) - train_y[-train_split])^2)
#   error_fnn_train = mean((test_predictions - train_y[-train_split])^2, na.rm = T)
#   
#   # Checking error
#   if(error_fnn_train < min_error_fnn){
#     
#     # Setting up test data
#     reshaped_data_tensor_test_final = array(dim = c(nrow(pre_test), ncol(pre_test), 1))
#     reshaped_data_tensor_test_final[, , 1] = as.matrix(pre_test)
#     
#     # Predictions
#     pred_fnn <- model_fnn %>% predict(pre_test)
#     # pred_fnn <- model_fnn %>% predict(reshaped_data_tensor_test_final)
#     
#     # Error
#     error_fnn = mean((c(pred_fnn) - test_y)^2, na.rm = T)
#     
#     # Saving training plots
#     fnn_training_plot[[i]] = as.data.frame(history_fnn)
#     
#     # New Min Error
#     min_error_fnn = error_fnn_train
#     
#   }
#   
# }

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