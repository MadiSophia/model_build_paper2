#packages
library(terra)
library(dplyr)
library(sf)
library(tidyr)


# clear out old data
rm(list = ls())
setwd("D:/paper2/")


# load data 
#lidar predictions

lidar <- read.csv("D:/paper2/high_low_rasters/1_31_clean_model/site_match/z05_model16.csv")


#read in variable rasters from beast outputs

variable <- list.files("Input_test/", pattern = ".tif$", full.names = T)
name <- list.files("Input_test/", pattern = ".tif$", full.names = F)

var_rast <- rast(variable)
name_1 <- unlist(name)




remove <- function(name) {
  substring(name, 1, nchar(name) - 4)  # Keep characters from the start up to the last 3
}

# Apply the function to the list of names
short <- sapply(name_1, remove)

#get rid of 2021 chane to generic inputs 

# Remove the middle number using gsub()
short_1 <- sub("-\\d{4}-", "-", short)
names(var_rast) <- short_1


#lidar <- lidar %>% drop_na(lat,lon)

# get lidar sample points 
points <- st_as_sf(lidar, coords = c("x", "y"), crs = 26910)



#extact BEAST values 
take <- vect(points$geometry)


extract <- take %>% project(var_rast)

beast_val <- terra::extract(var_rast, extract)

lidar$ID <- c(1: as.numeric(length(lidar$cover_2m)))
#Merge data.frams

lidar_beast <- merge(lidar, beast_val, by = "ID", all = TRUE)


column_names <- names(lidar_beast)

#"UTM_10S_elev_mean_2019", "UTM_10S_elev_p95_2019", "UTM_10S_percentage_first_returns_above_mean_2019" , 
#"UTM_10S_percentage_first_returns_above_2m_2019"

clean_data <-  lidar_beast[, c("x", "y", "beast", "disturbed", "cover_2m", "cover_mean",
                                                  "cover_2m_c", "cover_mean_c", "p_mean_c", "subbec",
                                         "p95_c", "p25_c", "p50_c", "sd_c", "rumple_c","cov_c", "density","p_diff", "amp-nbr", "amp-ndvi", "amp-tca", 
                                         "amp-tcb", "amp-tcg", "amp-tcw", "slp-nbr", "slp-ndvi", "slp-tca", "slp-tcb", 
                                         "slp-tcg", "slp-tcw", "trn-nbr", "trn-ndvi", "trn-tca", "trn-tcb", "trn-tcg", "trn-tcw")]

#pred 3 is the filtered 2020 values 

write.csv(clean_data, "D:/paper2/high_low_rasters/1_31_clean_model/model_data/model_data17.csv")

