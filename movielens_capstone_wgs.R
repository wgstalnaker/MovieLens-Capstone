
########################################################################
# MovieLens Recommendation System for detailed steps leading to 
# development see related report and Rmarkdown file. This R Script
# includes the minumum requirementsto ingest, manipulate, train and 
# test model, and product the final root mean square error calculation
########################################################################

######################################
# Create edx set, validation set
######################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###########################################
# Data Exploration 
###########################################
# structure and summary
head(edx)
str(edx)
summary(edx)
# check for null values
sapply(edx, {function(x) any(is.na(x))})
# unique counts
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId),
            n_genres = n_distinct(genres),
            n_rating = n_distinct(rating))
# Count of Movie Reviews by User
edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  labs(x = "User ID" , y = "Review Count") +
  geom_bar(color="black") + 
  scale_x_log10() + 
  ggtitle("Movie Reviews per User")

# Count of User Reviews by Movie
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  labs(x = "Movie ID", y = "Review Count") +
  geom_bar(color="black") + 
  scale_x_log10() + 
  ggtitle("User Reviews per Movie")

######################################
# Format source data timestamp and 
# extract and store year rated 
######################################
edx <- edx %>% 
  mutate(timestamp = as.POSIXct(timestamp, origin = "1960-01-01", tz = "GMT"))
edx <- edx %>% 
  mutate(year_rated = as.numeric(str_sub(timestamp, 1, 4)))
validation <- validation %>% 
  mutate(timestamp = as.POSIXct(timestamp, origin = "1960-01-01", tz = "GMT"))
validation <- validation %>% 
  mutate(year_rated = as.numeric(str_sub(timestamp, 1, 4)))
###########################################
# Extract year released from title 
###########################################
edx <- edx %>% 
  mutate(year_released = as.numeric(str_sub(title,-5, -2)))
validation<- validation %>% 
  mutate(year_released = as.numeric(str_sub(title,-5, -2)))
###########################################
# Flatten Genre data
###########################################
edx <- edx %>% separate_rows(genres, sep = "\\|")
validation <- validation %>% separate_rows(genres, sep = "\\|")

###################################################################################
# RMSE Function - The root mean squared error is defined as n is the number of
# users movie combinations and the sum is occruing over all combinations.
##################################################################################
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
###############################
# Regularized Movie + Genres + Year Released + User + Year Rated Effect Model"
###############################
# using cross validation to select penalty
penalty <- seq(10, 20, 0.25)
final_rmses <- sapply(penalty, function(p){
  mu <-mean(edx$rating)
  # movie effect with regularirzation
  m_e <- edx %>%
    group_by(movieId) %>%
    summarize(m_e = sum(rating - mu)/(n()+p), .groups = 'drop')
  # genre effect with regularirzation
  g_e <- edx %>% 
    left_join(m_e, by = "movieId") %>%
    group_by(genres) %>% 
    summarize(g_e = sum(rating - mu - m_e)/(n()+p), .groups = 'drop')   
  # year released effect with regularirzation
  y_e1 <- edx %>% 
    left_join(m_e, by = "movieId") %>%
    left_join(g_e, by="genres") %>%
    group_by(year_released) %>% 
    summarize(y_e1 = sum(rating - mu - m_e - g_e)/(n()+p), .groups = 'drop')
  # user effect with regularirzation
  u_e <- edx %>% 
    left_join(m_e, by = "movieId") %>%
    left_join(g_e, by = "genres") %>%
    left_join(y_e1, by = "year_released") %>% 
    group_by(userId) %>% 
    summarize(u_e = sum(rating - mu - m_e - g_e - y_e1)/(n()+p), .groups = 'drop') 
  # year rated effect with regularirzation
  y_e2 <- edx %>% 
    left_join(m_e, by = "movieId") %>%
    left_join(g_e, by = "genres") %>%
    left_join(y_e1, by = "year_released") %>% 
    left_join(u_e, by = "userId") %>% 
    group_by(year_rated) %>% 
    summarize(y_e2 = sum(rating - mu - m_e - g_e - y_e1 - u_e)/(n()+p), .groups = 'drop')    
  # predictions
  predicted_ratings <- validation %>%
    left_join(m_e, by = "movieId") %>%
    left_join(g_e, by = "genres") %>% 
    left_join(y_e1, by = "year_released") %>% 
    left_join(u_e, by = "userId") %>%
    left_join(y_e2, by = "year_rated") %>% 
    mutate(pred = mu + m_e + g_e + y_e1 + u_e + y_e2) %>%
    .$pred
  # Test the predicted results using the loss function
  return(RMSE(predicted_ratings, validation$rating))
})
# Create table for storing the final results 
final_rmse_results <- tibble(method = "Regularized Movie + Genres + Year Released + User + Year Rated Effect Model",  
                       RMSE = min(final_rmses))
# Print final results to the console
final_rmse_results %>% knitr::kable()
# review penalty
qplot(penalty, final_rmses)



