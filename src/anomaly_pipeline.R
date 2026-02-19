# =========================
# Data Mining - Assignment 1 - Task 1
# Weather anomaly detection
# (1) Density-based: kNN distance (k=5,15,30)
# (2) Reconstruction-based: Autoencoder (bottleneck=1,2,3) using keras3
# =========================

# ---- 0) Setup ----
set.seed(123)

# Put HW2023.csv in the same folder as this script
setwd("/Users/chendahua/Desktop/Data Mining/Assignment1")

# ---- 1) Load data ----
hw <- read.csv("HW2023.csv", header = TRUE)

# ---- 2) Cleaning ----
# Replace missing rain with 0
hw$rain[is.na(hw$rain)] <- 0

# Log transform rain to reduce skew
rain_log <- log1p(hw$rain)

# Select features for anomaly detection
# (Use rain_log instead of raw rain)
hw_feat <- hw[, c("cloudiness", "humidity", "wind_speed", "visibility", "temp_max")]
hw_feat$rain_log <- rain_log

# Standardize to remove scale differences
hw_scaled <- scale(hw_feat)

# Convert date for plotting
hw$date <- as.Date(hw$date, format = "%m/%d/%Y")

# =========================
# 3) Density-based anomaly (kNN distance)
# OLS_k = distance to k-th nearest neighbor
# =========================
dist_mat <- dist(hw_scaled)
dist_mat_full <- as.matrix(dist_mat)

# --- k = 5 ---
k <- 5
ols_k5 <- apply(dist_mat_full, 1, function(x) {
  sort(x)[k + 1]  # +1 because the smallest distance is 0 to itself
})
hw$OLS_k5 <- ols_k5

hw_sorted_k5 <- hw[order(-hw$OLS_k5), ]
top4_k5 <- hw_sorted_k5[1:4, ]
bottom2_k5 <- tail(hw_sorted_k5, 2)

# --- k = 15 ---
k <- 15
ols_k15 <- apply(dist_mat_full, 1, function(x) {
  sort(x)[k + 1]
})
hw$OLS_k15 <- ols_k15

hw_sorted_k15 <- hw[order(-hw$OLS_k15), ]
top4_k15 <- hw_sorted_k15[1:4, ]
bottom2_k15 <- tail(hw_sorted_k15, 2)

# --- k = 30 ---
k <- 30
ols_k30 <- apply(dist_mat_full, 1, function(x) {
  sort(x)[k + 1]
})
hw$OLS_k30 <- ols_k30

hw_sorted_k30 <- hw[order(-hw$OLS_k30), ]
top4_k30 <- hw_sorted_k30[1:4, ]
bottom2_k30 <- tail(hw_sorted_k30, 2)

# Plot density-based scores (optional but good for report)
plot(hw$date, hw$OLS_k15,
     type = "l",
     xlab = "Date",
     ylab = "Anomaly Score (k = 15)",
     main = "Density-based Anomaly Scores over 2023")

# =========================
# 4) Reconstruction-based anomaly (Autoencoder, keras3)
# OLS_AE = mean squared reconstruction error
# =========================
suppressPackageStartupMessages(library(keras3))

x_train <- as.matrix(hw_scaled)
input_dim <- ncol(x_train)

# -------------------------
# AE setting 1: bottleneck = 1
# -------------------------
model_ae1 <- keras_model_sequential() |>
  layer_dense(units = 4, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 1, activation = "relu") |>
  layer_dense(units = 4, activation = "relu") |>
  layer_dense(units = input_dim)

model_ae1 |> compile(optimizer = optimizer_adam(), loss = "mse")

model_ae1 |> fit(
  x = x_train, y = x_train,
  epochs = 100, batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

x_recon1 <- model_ae1 |> predict(x_train)
hw$OLS_AE_1 <- rowMeans((x_train - x_recon1)^2)

hw_sorted_ae1 <- hw[order(-hw$OLS_AE_1), ]
show_cols_ae <- c("date","temp_max","humidity","visibility","cloudiness","wind_speed","rain")
top4_ae1 <- hw_sorted_ae1[1:4, c(show_cols_ae, "OLS_AE_1")]
bottom2_ae1 <- tail(hw_sorted_ae1, 2)[, c(show_cols_ae, "OLS_AE_1")]

# -------------------------
# AE setting 2: bottleneck = 2
# -------------------------
model_ae2 <- keras_model_sequential() |>
  layer_dense(units = 4, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 2, activation = "relu") |>
  layer_dense(units = 4, activation = "relu") |>
  layer_dense(units = input_dim)

model_ae2 |> compile(optimizer = optimizer_adam(), loss = "mse")

model_ae2 |> fit(
  x = x_train, y = x_train,
  epochs = 100, batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

x_recon2 <- model_ae2 |> predict(x_train)
hw$OLS_AE_2 <- rowMeans((x_train - x_recon2)^2)

hw_sorted_ae2 <- hw[order(-hw$OLS_AE_2), ]
top4_ae2 <- hw_sorted_ae2[1:4, c(show_cols_ae, "OLS_AE_2")]
bottom2_ae2 <- tail(hw_sorted_ae2, 2)[, c(show_cols_ae, "OLS_AE_2")]

# Plot AE setting 2 (optional but good for report)
png("AE_bottleneck_2.png", width = 800, height = 500)
plot(hw$date, hw$OLS_AE_2,
     type = "l",
     xlab = "Date",
     ylab = "AE OLS (bottleneck=2)",
     main = "Autoencoder Anomaly Scores over 2023")
dev.off()

# -------------------------
# AE setting 3: bottleneck = 3
# -------------------------
model_ae3 <- keras_model_sequential() |>
  layer_dense(units = 4, activation = "relu", input_shape = input_dim) |>
  layer_dense(units = 3, activation = "relu") |>
  layer_dense(units = 4, activation = "relu") |>
  layer_dense(units = input_dim)

model_ae3 |> compile(optimizer = optimizer_adam(), loss = "mse")

model_ae3 |> fit(
  x = x_train, y = x_train,
  epochs = 100, batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

x_recon3 <- model_ae3 |> predict(x_train)
hw$OLS_AE_3 <- rowMeans((x_train - x_recon3)^2)

hw_sorted_ae3 <- hw[order(-hw$OLS_AE_3), ]
top4_ae3 <- hw_sorted_ae3[1:4, c(show_cols_ae, "OLS_AE_3")]
bottom2_ae3 <- tail(hw_sorted_ae3, 2)[, c(show_cols_ae, "OLS_AE_3")]

# =========================
# 5) (Optional) Quick check outputs in Console
# =========================
# Print clean tables if you want to copy into report
print(top4_k5); print(bottom2_k5)
print(top4_k15); print(bottom2_k15)
print(top4_k30); print(bottom2_k30)

print(top4_ae1); print(bottom2_ae1)
print(top4_ae2); print(bottom2_ae2)
print(top4_ae3); print(bottom2_ae3)
