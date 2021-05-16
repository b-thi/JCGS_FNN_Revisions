##############################
# Tuning For Weather         #
##############################

# Example file illustrating tuning method

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
total_prec = apply(daily$precav, 2, mean)

# Creating functional data
temp_data = array(dim = c(65, 35, 1))
tempbasis65  = create.fourier.basis(c(0,365), 65)
timepts = seq(1, 365, 1)
temp_fd = Data2fd(timepts, daily$tempav, tempbasis65)

# Data set up
temp_data[,,1] = temp_fd$coefs

# Running FNN for weather (No Tuning)
fnn_weather = FNN(resp = total_prec, 
                  func_cov = temp_data, 
                  scalar_cov = NULL,
                  basis_choice = c("fourier"), 
                  num_basis = 11,
                  hidden_layers = 2,
                  neurons_per_layer = c(32, 32),
                  activations_in_layers = c("relu", "sigmoid"),
                  domain_range = list(c(1, 365)),
                  epochs = 250,
                  output_size = 1,
                  loss_choice = "mse",
                  metric_choice = list("mean_squared_error"),
                  val_split = 0.2,
                  patience_param = 25,
                  learn_rate = 0.1,
                  early_stop = T,
                  print_info = F)

# THIS IS JUST RANDOM SET UP ABOVE.
# This turns out to be the 120th combination and the MSPE is available below. The goal was to see how much improvement we can get 
# after tuning

# THIS FILE IS JUST AN EXAMPLE OF HOW TUNING WOULD WORK IN THE PACKAGE
# THE GRID FOR HOW WE GOT THE PARAMETERS IN OUR WEATHER RESULTS WAS BIGGER
# This was for the purposes of trying to demonstrate the tuning process

# Creating grid
tune_list_weather = list(num_hidden_layers = c(2),
                      neurons = c(8, 16, 32),
                      epochs = c(250),
                      val_split = c(0.2),
                      patience = c(15),
                      learn_rate = c(0.01, 0.05, 0.1),
                      num_basis = c(5, 7, 11),
                      activation_choice = c("relu", "sigmoid"))

# What the grid looks like:

# Current layer number
current_layer = tune_list_weather$num_hidden_layers[1]

# Creating data frame of list
df = expand.grid(rep(list(tune_list_weather$neurons), tune_list_weather$num_hidden_layers[1]), stringsAsFactors = F)
df2 = expand.grid(rep(list(tune_list_weather$num_basis), 1), stringsAsFactors = F)
df3 = expand.grid(rep(list(tune_list_weather$activation_choice), tune_list_weather$num_hidden_layers[1]), stringsAsFactors = F)
colnames(df2)[1] <- "Var2.y"
colnames(df3)[1] <- "Var2.z"

# Getting grid
pre_grid = expand.grid(df$Var1,
                       Var2.y = df2$Var2.y,
                       Var2.z = df3$Var2.z,
                       tune_list_weather$epochs,
                       tune_list_weather$val_split,
                       tune_list_weather$patience,
                       tune_list_weather$learn_rate)

# Merging
combined <- unique(merge(df, pre_grid, by = "Var1"))
combined2 <- unique(merge(df2, combined, by = "Var2.y"))
final_grid <- suppressWarnings(unique(merge(df3, combined2, by = "Var2.z")))

# Running Tuning
Weather_Tuned = FNN_Tune(tune_list_weather,
                         range_01(total_prec), 
                         temp_data,
                         basis_choice = c("fourier"),
                         domain_range = list(c(1, 24)),
                         nfolds = 2,
                         cores = 4)

# Looking at results
tail(Weather_Tuned$Grid_List[[1]])
Weather_Tuned$Parameters
Weather_Tuned$All_Information[[1]][[120]]

# Check 1
# Check 2