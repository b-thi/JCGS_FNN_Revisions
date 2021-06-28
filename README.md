## Files for: Deep Learning with Functional Inputs (Revision)

---------------------------------------------------------------

### This text file has information on each script for the paper "Deep Learning with Functional Inputs". It also has information on what package versions are required to make sure that they all run

---------------------------------------------------------------

- FNN.R = The main function file containing the backbone for every script. It contains the core functions for FNNs and also has great overlap with the FuncNN package. We use these functions as opposed to the associated package because the package was developed after this paper was initially submitted.

- FNN_Figure2.R = This script produces Figure 2 in the manuscript.

- FNN_AblatioStudies.R = This script produces the Ablation results found in the supplementary document.

- FNN_BikeExample.R = This script contains all the main results for the bike example in the manuscript (Found in Section 3.1).

- FNN_TecatorExample.R = This script contains all the results for the tecator example in the manuscript (Found in Section 3.2).

- FNN_WeatherExample.R = This script contains all the main results for the weather example in the manuscript (Found in Section 3.3).

- FNN_FunctionalWeights.R = This script produces the Functional Weight examples seen in the Bike and Weather real data examples (Found in Section 3.1, 3.3).

- FNN_RecoverySimulation.R = This script contains the results for recovery beta(t) in Section 4.1.

- FNN_PredictionSimulation.R = This script contains the simulation predictions in Section 4.2.

- PackageVersions.lock = Contains the exact versions of all the packages loaded in FNN.R. This is for reproducibility purposes (see below) and this file opens in R.

- bike.RData = Data for the bike example.

- tecator.RDS = Data for the tecator example.

- daily.RDS = Data for the weather example.

---------------------------------------------------------------

### Since we are using Keras/Tensorflow in the background, an installation of Python is required on the machine. Our results also require particular versions of these so, to make sure things work, we recommend the following steps (Windows):

1. Download Anaconda (https://www.anaconda.com/products/individual)

2. Open Anaconda Prompt

3. Create Python 3.7 environment
3a. conda create --name python37 activate

4. Install the Python version so Keras and Tensorflow 
4a. pip install tensorflow==1.14.0 
4b. pip install keras==2.2.4

5. Now, we need to open R and install the correct versions of Keras and Tensorflow for R 
5a. install_version("tensorflow", version = "2.2.0", repos = "http://cran.us.r-project.org") 
5b. install_version("keras", version = "2.2.5.0", repos = "http://cran.us.r-project.org")

6. R wil likely default to the main version of Python you install when you installed Anaconda so you need to set it to the virtual environment we created earlier using Reticulate
6a. library(reticulate)
6b. use_condaenv(condaenv = 'PFDA', conda = "C:/~path_to_version/anaconda3/envs/Python37/python.exe")
6c. use_python("C:/~path_to_version/anaconda3/envs/Python37/python.exe")
6d. As a side point, remember to delete .Rdata in your working directory and restart the R session before setting the above paths (otherwise R will default to that information)

7. At this point, everything should be working and you can just confirm you have the right versions of all the packages by looking at the PackageVersions.Lock file!

If there are any issues, feel free to contact us (you may also need to install the r-versions [not to be confused with R] of Keras and Tensorflow in Anaconda as well which has the versions r-keras 2.2.4.1 & r-tensorflow 1.13.1, respectively.) and we
will attempt to respond to you promptly. We apologize for any inconvenience but this is a neccesary evil for anyone running these infrastructures in an R environment! Thank you.