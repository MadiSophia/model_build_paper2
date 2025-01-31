library(dplyr)
library(arrow)
library(tictoc)
library(tidyr)
library(cowplot)

rm(list =ls())

#undisturbed areas in 2021
u <- read.csv("D:/paper2/high_low_rasters/update_structure_csv/csv/high_u2.csv")

#disturbed areas 2021
#do a site match for a random sample 
d <- read.csv( "D:/paper2/high_low_rasters/update_structure_csv/csv/rand_d2.csv")

################################################################################
#subsetc my 2019 structural metrics
#ensure undisturbed pixels 
u$subbec <- paste(u$SUBZONE, u$ZONE, sep = "-")
d$subbec <- paste(d$SUBZONE, d$ZONE, sep = "-")

#############################################################################
#for undisturbed pixel we want eliminate low points take 90% canopy cover by subbec zone 
# by canopy cover 

filtered_df <- u %>%
  group_by(subbec) %>%
  filter( elev95 > quantile(elev95, 0.9, na.rm = TRUE))  %>%
  ungroup()
###############################################################################


##################################################################################

# View the filtered dataset
print(filtered_df)


##########
#add some base line undisturbed samples for training data purposes 
set.seed(42)  
u_train <- filtered_df[sample(1:nrow(filtered_df), 5000), ]

# Remove these samples from filtered_df
filtered_df <- filtered_df[-as.numeric(rownames(u_train)), ]
#write.csv(filtered_df, "D:/paper2/quick_test/test_df.csv")
############################################################################
#1 we want to only look at shared BEC and sub bec zones
#we are building our model of 10,000 disturbed points so we will filter these down too 
d_new <- d[sample(1:nrow(d), 7000), ]

d_new <- rbind(d_new, u_train)

d_new$disturbed <- "Yes"

bec <- unique(d_new$ZONE)
bsub <- unique(d_new$SUBZONE)

#we want only undisturbed sites with same bec and subbeczones
u_new <- filtered_df[filtered_df$ZONE %in% bec, ]
u_new <- filtered_df[filtered_df$SUBZONE %in% bsub, ]
###############################################################################
#normalize height values from NTEMS real quick 
d_new$pred <- 1
u_new$pred <- 0 


shared <- names(d_new)

u_new <- u_new %>% select(shared)


#combine data 
data <- rbind(u_new, d_new)



#calculate beta 
beta <- glm(pred ~ slope + dtm + aspect + MAP + MAT + MCMT + MWMT,   data = data)

# now calculate propensity scores
data$p_raw <- as.numeric(predict(beta, type = "response"))


###############################################################################
#3 Now we want to get rid of any lidar data that is messy or incorrect 
#we don't want any negative height of elevation variables

# Columns to check for negative values
check <- c("p25", "cover_2m", "p50", "elev95", "elev_mean", "cover_mean")

# Filter out rows with negative values in specific columns
data_clean <- data %>%
  filter(across(all_of(check), ~ . >= 0))


#########################################################
#4 We are going to seperate back out our disturbed and undisturbed samples 
# Add new variables for site selection 
# make change variables


d_model <- subset(data_clean, disturbed == "Yes")
#d_model <- d_model[sample(1:nrow(d_model), 1000), ]
u_model <- subset(data_clean, disturbed == "No")


# new variables
d_model$cover_2m_c <- NA
d_model$cover_mean_c <- NA
d_model$p_mean_c <- NA
d_model$p95_c <- NA
d_model$p25_c <- NA
d_model$p50_c <- NA
d_model$sd_c <- NA
d_model$rumple_c <- NA
d_model$cov_c <- NA
d_model$density <-NA
d_model$index <- NA
d_model$DAPS <-  NA
 d_model$dist <- NA
 d_model$p_diff <- NA
 

#add indexing to to u so we know which pixles are picked
u_model <- u_model %>%
  mutate(index = paste0('u', seq_len(nrow(u_model))))

###############################################################################
#5 now we pick our sites using a distance weighted propensity socre 
i <- 14047

tic()
for(i in 1:nrow(d_new)){
  #filter by bec zone
  p <- d_model[i,]
  
  #filter by BEC and sub beczone 
  site <- subset(u_model, ZONE == p$ZONE)
  site <- subset(site, SUBZONE == p$SUBZONE) 
  
  #now we calculate the difference for points 
  site$dist <- NA
  site$p_diff <- NA
  site$DAPS <- NA
  
  #calculate variables
  site$dist <- sqrt(((p$x - site$x)^2) + ((p$y - site$y)^2))
  site$p_diff <-  abs(site$p_raw - p$p_raw)
  
  
  # maybe attemp threshold on pdiff 
  
  
  #calculate distance weighted to propesity score using from Woo et al. (2021)
  w <- 1

  #looked best with no site difference 
  
  
  #attempt with removing outlier for poorly matched areas
  site$DAPS <- (w* site$p_diff ) + ((1- w)*(site$dist))
  
 
  #now select point with minimal site difference 
  o <- subset(site, site$DAPS ==   min(na.omit(site$DAPS)))  
  
  # add flag to select one point o if distance are the same
  if(nrow(o)> 1){o <- o[sample(nrow(o), 1), ]} else{ o == o}
  
  # calculate disturbance change for lidar metric
  d_model$cover_2m_c[i] <- p$cover_2m - o$cover_2m
  d_model$cover_mean_c[i] <- p$cover_mean - o$cover_mean
  d_model$p_mean_c[i] <- p$elev_mean - o$elev_mean
  d_model$p95_c[i] <- p$elev95 - o$elev95
  d_model$p25_c[i] <- p$p25 - o$p25
  d_model$p50_c[i] <- p$p50 - o$p50
  d_model$sd_c[i] <- p$sd - o$sd
  d_model$rumple_c[i] <-  p$rumple - o$rumple
  d_model$cov_c[i] <-  p$cov - o$cov
  d_model$density[i] <-p$desity - o$desity
  d_model$index[i] <- o$index
  d_model$DAPS[i] <- o$DAPS
  d_model$dist[i] <- o$dist
  d_model$p_diff[i] <- o$p_diff
  

  
  
}
condtoc()
################################################################################
#removing mad matchers and outliers 
# Example: Removing outliers using the modified Z-score method
# Function to calculate modified Z-scores


ggplot(d_model, aes(x = p_diff)) +
  geom_histogram(binwidth = 0.0008, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Histogram of p_diff", x = "p_diff", y = "Count")




modified_zscore <- function(x) {
  median_x <- median(x)
  mad_x <- mad(x)
  return(0.6745 * (x - median_x) / mad_x)
}


# Calculate modified Z-scores for the 'p_diff' column
z_scores <- modified_zscore(d_model$p_diff)

# Define outlier threshold (commonly 3.5)
outliers <- d_model$p_diff[abs(z_scores) > 2]


#Identify rows with outliers (Z-score > 3.5)
outlier_rows <- which(abs(z_scores) > 2)

# Remove outliers from data
model_no_outliers <- d_model[-outlier_rows, ]

ggplot(model_no_outliers, aes(x = p_diff)) +
  geom_histogram(binwidth = 0.0008, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Histogram of p_diff", x = "p_diff", y = "Count")

#################################################################################
# List of columns to filter
################################################################################

columns_to_filter <- c("cover_2m_c", "cover_mean_c", "p_mean_c", "p95_c", 
                       "p25_c", "p50_c", "sd_c", "rumple_c", "cov_c", "density")

# Create a copy of your dataset
filtered_d_model_no_outliers <- model_no_outliers

# Loop through each column and filter based on thresholds
#for (col in columns_to_filter) {
  # Calculate the thresholds for the bottom and top 1%
  #lower_threshold <- quantile(filtered_d_model_no_outliers[[col]], 0.01, na.rm = TRUE)
  #upper_threshold <- quantile(filtered_d_model_no_outliers[[col]], 0.99, na.rm = TRUE)
  
  # Filter rows for the current column
  #filtered_d_model_no_outliers <- filtered_d_model_no_outliers[
    #filtered_d_model_no_outliers[[col]] > lower_threshold & 
      #filtered_d_model_no_outliers[[col]] < upper_threshold, 
  #]
#}



#################################################################################



write.csv(model_no_outliers , "D:/paper2/high_low_rasters/1_31_clean_model/site_match/z05_model17.csv") 


write.csv(u_model, "D:/paper2/high_low_rasters/1_31_clean_model/site_match/z05_umodel17.csv")

unique(filtered_d_model_no_outliers$index)

###############################################################

# Assuming your dataset is named 'site' and the column of interest is 'p_diff'







