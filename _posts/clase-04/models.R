library(USgas)
library(dplyr)
library(parsnip)
library(rsample)
library(timetk)
library(modeltime)

# The us_monthly dataset provides a monthly time series, 
# representing the demand for natural gas in the US between 2001 and 2020:
data("us_monthly")
head(us_monthly)

splits <- us_monthly %>% 
  timetk::tk_tbl() %>% 
  initial_time_split(prop = 0.8)

# SES
model_spec_ses <- exp_smoothing( # SES
  error = "additive",
  trend = "none",
  seasonal = "none"
) %>%
  set_engine("ets")

model_fit_ses <- model_spec_ses %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_ses

# HOLT
model_spec_holt <- exp_smoothing( # SES
  error = "additive",
  trend = "additive",
  seasonal = "none"
) %>%
  set_engine("ets")

model_fit_holt <- model_spec_holt %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_holt

# HOLT DAMPED
model_spec_holt_damped <- exp_smoothing( # SES
  error = "additive",
  trend = "additive",
  seasonal = "none",
  damping = "damped"
) %>%
  set_engine("ets")

model_fit_holt_damped  <- model_spec_holt_damped  %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_holt_damped

# HOLT WINTERS aditivo
model_spec_hw <- exp_smoothing(
  error = "additive",
  trend = "additive",
  season = "additive",
  seasonal_period = 12
) %>%
  set_engine("ets")

model_fit_hw  <- model_spec_hw  %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_hw

# HOLT WINTERS multiplicativo
model_spec_hw_m <- exp_smoothing(
  error = "multiplicative",
  trend = "multiplicative",
  season = "multiplicative",
  seasonal_period = 12
) %>%
  set_engine("ets")

model_fit_hw_m  <- model_spec_hw_m  %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_hw_m

# HOLT WINTERS multiplicativo con damping
model_spec_hw_m_damped <- exp_smoothing(
  error = "multiplicative",
  trend = "multiplicative",
  season = "multiplicative",
  damping = "damped",
  seasonal_period = 12
) %>%
  set_engine("ets")

model_fit_hw_m_damped  <- model_spec_hw_m_damped  %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_hw_m_damped

# LM
model_spec_lm <- linear_reg() %>%
  set_engine("lm")

model_fit_lm <- model_spec_lm %>%
  fit(
    y ~ as.numeric(date) + I(as.numeric(date) ^ 2) + lubridate::month(date, label = TRUE),
    data = training(splits)
  )

# SNAIVE

model_spec_snaive <- naive_reg(seasonal_period = 12) %>%
  set_engine("snaive")

model_fit_snaive <- model_spec_snaive %>%
  fit(y ~ date , data = training(splits))
model_fit_snaive

# PROPHET
model_spec_prophet <- prophet_reg() %>%
  set_engine("prophet")

model_fit_prophet <- model_spec_prophet %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_prophet

# TBATS
model_spec_tbats <- seasonal_reg() %>%
  set_engine("tbats")

model_fit_tbats <- model_spec_tbats %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_tbats

# ARIMA
model_spec_arima <- arima_reg() %>%
  set_engine("auto_arima")

model_fit_arima <- model_spec_arima %>%
  fit(
    y ~ date,
    data = training(splits)
  )

model_fit_arima

# Linear Regression with ARIMA errors

model_fit_linreg_arima <- model_spec_arima %>% 
  fit(
    y ~ date
    + lubridate::month(date, label = TRUE)
    + fourier_vec(date, period = 12)
    + fourier_vec(date, period = 24)
    + fourier_vec(date, period = 48),
    data = training(splits)
  )

# MODELTIME TABLE

models_tbl <- modeltime_table(
  model_fit_ses,
  model_fit_holt,
  model_fit_holt_damped,
  model_fit_hw,
  model_fit_hw_m,
  model_fit_hw_m_damped,
  model_fit_prophet,
  model_fit_tbats,
  model_fit_arima,
  model_fit_linreg_arima,
  model_fit_lm,
  model_fit_snaive
)

models_tbl

calibration_tbl <- models_tbl %>%
  # Calibracion en datos de test para evaluacion de rendimiento
  modeltime_calibrate(new_data = testing(splits)) %>%
  # Pronostico en datos de test
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = us_monthly
  )
calibration_tbl

calibration_tbl %>% 
  # Grafico pronosticos
  plot_modeltime_forecast(.interactive = TRUE)

# Metricas de rendimiento
models_tbl %>%
  # Calibracion en datos de test para evaluacion de rendimiento
  modeltime_calibrate(new_data = testing(splits)) %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = TRUE
  )
