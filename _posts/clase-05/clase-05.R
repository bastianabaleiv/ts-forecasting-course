# Manipulacion de series de tiempo
library(timetk) # https://business-science.github.io/timetk/

# Paquetes de Data Science en R
# dplyr - ggplot - forcats - tibble - readr - stringr - tidyr - purrr
library(tidyverse) # https://dplyr.tidyverse.org/

# Coleccion de paquetes para modelamiento y Machine Learning usando principios tidy
library(tidymodels) # https://www.tidymodels.org/

# Framework para aplicar modelos de machine learning y series de tiempo
library(modeltime) # https://business-science.github.io/modeltime/

# Coleccion de paquetes para modelamiento y Machine Learning
# caret:Classification And REgression Training 
library(caret) # https://topepo.github.io/caret/data-splitting.html#time

# Framework de modelos para series de tiempo
library(forecast)

# Graficos interactivos
library(plotly)

# Wrappers para modelso basados en reglas 
library(rules)

# Timing
library(tictoc)

# Procesamiento en paralelo
library(future)
library(doFuture)

# Data
library(USgas) # https://cran.r-project.org/web/packages/USgas/index.html
# Data
data("us_monthly")

# Grafico 
us_monthly_plot <- us_monthly %>%
  ggplot(aes(x = date, y = y)) +
  geom_line() +
  theme_bw()

ggplotly(us_monthly_plot)

# Old-school partitioning: caret ------------------------------------------

#?createTimeSlices
# Muestreo para datasets con fuerte componente temporal
# Las muestras no son aleatorias y contienen valores consecutivos en el tiempo
# La funcion asume que los datos estan ordenados de manera cronologica
# createTimeSlices(y, initialWindow, horizon = 1, fixedWindow = TRUE, skip = 0)
caret_split <- caret::createTimeSlices(
  # Serie de tiempo como vector, ordenada de manera cronologica
  y = us_monthly$y,
  # Cantidad inicial de valores consecutivos en cada conjunto de entrenamiento
  initialWindow = 12 * 19, # 12 meses x 19 anios
  # meses x anios
  # Cantidad de valores consecutivos en cada conjunto de evaluacion
  horizon = 3,
  # ~ horizonte de pronostico
  # Acumulacion de datos
  # deberia crecer el tamanio muestral de los datos de entrenamiento a traves de
  # las particiones?
  fixedWindow = FALSE,
  # cuantas observaciones omitir (saltar) entre datos de entrenamiento y evaluacion
  skip = 2
)

# Estructura lista con particiones
str(caret_split)

# Lista a poblar con pronosticos
forecast_list <-
  vector(mode = "list", length = length(caret_split$train))

# Lista de modelos
model_list <-
  vector(mode = "list", length = length(caret_split$train))

# Para cada particion
for (i in 1:length(caret_split$train)) {
  
  # Ajustamos un modelo ARIMA (especificacion automatica)
  fitted_arima <- us_monthly[caret_split$train[[i]], ] %>% 
    # Transformamos ts
    timetk::tk_ts(
      start = .$date[[1]] %>% zoo::as.yearmon(),
      frequency = 12,
      silent = TRUE
    ) %>% 
    # Ajustamos segun algoritmo auto.arima
    forecast::auto.arima(seasonal = TRUE)
  
  # Guardamos el modelo
  model_list[[i]] <- fitted_arima
  
  # Generamos pronostico a tres meses
  forecast_arima <- forecast::forecast(fitted_arima, h = 3)
  
  # Guardamos datos junto a datos de evaluacion
  forecast_list[[i]] <-
    cbind(us_monthly[caret_split$test[[i]], ], yhat = forecast_arima$mean)
  
}

# Unimos los pronosticos generados
forecast_tbl <- forecast_list %>% bind_rows() %>% as_tibble()

# Graficamos
forecast_tbl %>%
  dplyr::rename(actual = y,
                forecast = yhat) %>%
  # Pivoteo
  tidyr::pivot_longer(cols = -date,
                      names_to = "Series",
                      values_to = "Value") %>%
  ggplot(aes(x = date, y = Value, color = Series)) +
  geom_line() +
  geom_point() +
  theme_classic()

# Evaluacion del modelo
forecast_accuracy <-
  tibble(
    # RMSE
    rmse = Metrics::rmse(actual = forecast_tbl$y, predicted = forecast_tbl$yhat),
    # MAE
    mae = Metrics::mae(actual = forecast_tbl$y, predicted = forecast_tbl$yhat),
    # MAPE
    mape =  Metrics::mape(actual = forecast_tbl$y, predicted = forecast_tbl$yhat) *
      100
  )

# Tidymodels approach -----------------------------------------------------
# https://www.tidymodels.org/learn/models/time-series/
roll_rs <- us_monthly %>%
  rsample::rolling_origin(
    # Cantidad inicial de valores consecutivos en cada conjunto de entrenamiento
    initial = 12 * 19,
    # Cantidad de valores consecutivos en cada conjunto de evaluacion
    assess = 3,
    # ~ horizon, horizonte de pronostico
    # Acumulacion de datos
    cumulative = TRUE,
    # Cuantas observaciones omitir (saltar) entre datos de entrenamiento y evaluacion
    skip = 2,
    # Inclusion de rezagos entre set de entrenamiento y evaluacion, util al utilizar
    # predictores rezagados
    lag = 0
  )

# Obtenemos un tibble con las particiones y su identificador
roll_rs

# Verificamos su clase
class(roll_rs) # rolling_origin, rset

# Cada particion (split) contiene la informacion asociada a su respectiva muestra
roll_rs$splits[[1]]

# Podemos obtener los datos de entrenamiento con el metodo training()
training(roll_rs$splits[[1]]) # para la primera particion

# Y verificar que lo obtenido es lo mismo que con createTimeSlices() en entrenamiento
for (i in 1:nrow(roll_rs)) {
  if (identical(training(roll_rs$splits[[i]]), us_monthly[caret_split$train[[i]], ])) {
    print(paste0("Particion entrenamiento createTimeSlices/rolling_origin ", i, " idénticas"))
  }
}

# y en los datos de evaluacion
for (i in 1:nrow(roll_rs)) {
  if (identical(testing(roll_rs$splits[[i]]), us_monthly[caret_split$test[[i]], ])) {
    print(paste0("Particion evaluación createTimeSlices/rolling_origin ", i, " idénticas"))
  }
}

# La estructura de datos nos permite utilizar una aproximacion mas funcional al
# ajuste de modelos

# Creamos una columna con el modelo ajustado
roll_rs$arima <- 
  purrr:::map(roll_rs$splits, function(x){
    x %>% 
      # Extrae datos de entrenamiento
      training() %>% 
      timetk::tk_ts(
        start = .$date[[1]] %>% zoo::as.yearmon(),
        frequency = 12,
        silent = TRUE
      ) %>% 
      forecast::auto.arima(seasonal = TRUE)
  })

# Verificamos que los modelos son identicos
for (i in 1:nrow(roll_rs)) {
  if (identical(model_list[[i]], roll_rs$arima[[i]])) {
    print(paste0("Modelo partición ", i, " idénticos"))
  }
}

# Pronostico
roll_rs$yhat <- 
  purrr:::map(roll_rs$arima, function(x){
    forecast_arima <- x %>% 
      forecast::forecast(h = 3)
    
    forecast_arima$mean %>% timetk::tk_tbl()
  })

# Unir filas con pronosticos
roll_rs$yhat %>% 
  dplyr::bind_rows() %>% 
  mutate(date = lubridate::as_date(index)) %>% 
  # Eliminamos index
  select(-index) %>% 
  select(date,
         yhat = value)

# Dynamic Regression Model ------------------------------------------------
# Regression with ARIMA errors

us_monthly_augmented <- us_monthly %>%
  mutate(
    # Rezago estacional
    lag12 = dplyr::lag(y, n = 12),
    # Variable categorica que representa el componente estacional de la serie
    month = factor(lubridate::month(date, label = FALSE), ordered = FALSE),
    # tendencia
    trend = 1:n(),
    # tendencia polynomial
    poly_trend = trend^2
  )

us_monthly_augmented <- us_monthly_augmented %>% 
  select(-c(month)) %>% 
  cbind(., model.matrix(~month, us_monthly_augmented)) %>% 
  filter(complete.cases(.))

train_augmented <- us_monthly_augmented %>%
  filter(date < "2020-01-01") %>% 
  select(-date)

test_augmented <- us_monthly_augmented %>%
  filter(date >= "2020-01-01") %>% 
  select(-date)

# Ajustamos modelo de regresion dinamica
dynreg <- forecast::auto.arima(
  y = train_augmented[,1],
  xreg = as.matrix(train_augmented[, -1])
) 

forecast::ggtsdisplay(forecast::arima.errors(dynreg), main = "ARIMA errors")

forecast::checkresiduals(dynreg, main = "ARIMA residuals")

summary(dynreg)

# [!] Se generara pronostico como tantas filas tenga xreg
forecast_dynreg <- forecast::forecast(dynreg, xreg = as.matrix(test_augmented[,-1]))

forecast::autoplot(forecast_dynreg)

# Machine Learning approach -----------------------------------------------
# (modeltime ecosystem)

us_monthly_tbl <- us_monthly %>% 
  timetk::tk_tbl()

us_monthly_tbl %>% 
  plot_time_series(.date_var = date,
                   .value = y)

us_monthly_trans_tbl <- us_monthly_tbl %>% 
  # Log para controlar varianza TIP!
  mutate(y_log = log_interval_vec(y, limit_lower = 0, offset = 1),
         # Estandarizacion usual para modelos de ML
         y_log_std = standardize_vec(y_log))

# Graficamos
us_monthly_trans_tbl %>% 
  plot_time_series(.date_var = date,
                   .value = y_log_std)

# Particion entrenamiento y evaluacion
splits <- timetk::time_series_split(
  us_monthly_tbl,
  assess = "14 months",
  cumulative = TRUE
)

# Especificacion al estilo Machine Learning -------------------------------

# Feature Engineering 
# scikit-learn kinda like
recipe_spec <- recipes::recipe(y ~ ., data = training(splits)) %>%
  # Generar variables temporales
  timetk::step_timeseries_signature(date) %>%
  # Removemos columnas que al parecer no nos son de utilidad
  step_rm( # Tambien se puede usar "matches"
    ends_with("iso"),
    ends_with(".xts"),
    contains("hour"),
    contains("minute"),
    contains("seconds"),
    contains("am.pm")
  ) %>% 
  step_normalize(
    ends_with("index.num"),
    ends_with("_year"),
    contains("yday")
  ) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_fourier(date, period = c(3,12), K = 2)

recipe_spec %>% prep() %>% juice() %>% glimpse()

# Random forest
model_spec_rf <- parsnip::rand_forest(
  mtry = 10,
  trees = 2000
) %>% 
  set_engine("randomForest") %>% 
  set_mode("regression")

set.seed(123)

# Definimos workflow
wrkflw_fit_rf <- workflows::workflow() %>% 
  # Agregamos modelo a ajustar
  workflows::add_model(model_spec_rf) %>% 
  # Agregamos receta (feature engineering)
  workflows::add_recipe(recipe_spec) %>% 
  # Ajustamos a los datos de entrenamiento
  fit(training(splits))

# Agregamos a tabla de modelos el modelo ajustado
model_tbl <- modeltime::modeltime_table(
  wrkflw_fit_rf
)

# Calibramos con los datos de testeo
calibration_tbl <- model_tbl %>% 
  modeltime::modeltime_calibrate(testing(splits))

# Graficamos Random Forest
calibration_tbl %>% 
  modeltime::modeltime_forecast(
    new_data = testing(splits),
    actual_data = us_monthly_tbl
  ) %>% 
  plot_modeltime_forecast()

# Podemos revisar el desempenio del modelo en test rapidamente y ver si el modelo
# es prometedor
calibration_tbl %>% 
  modeltime_accuracy()

# Particiones de entrenamiento y test
resamples_tscv <- timetk::time_series_cv(
  data = training(splits),
  date_var = date,
  initial = 12*18,
  assess = 3,
  skip = 2,
  cumulative = TRUE
)

# Particiones para validacion cruzada
resamples_tscv

# Esquema grafico de validacion cruzada:
# Particiones de los datos de entrenamiento en subconjuntos de entrenamiento 
# y evaluacion para ajustar parametros sin utilizar los datos de evaluacion reales
resamples_tscv %>% 
  timetk::tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(
    .date_var = date,
    .value = y
  )

# Ajuste de hiperparametros (Hyperparameter tuning) -----------------------

# Revisamos hiper-parametros asociados al modelo
wrkflw_fit_rf %>% extract_spec_parsnip()

model_spec_rf <- rand_forest(
  mtry = tune(),
  trees = tune()
) %>% 
  set_engine("randomForest") %>% 
  set_mode("regression")

model_spec_rf

# Intento 1

?update

set.seed(123)

grid_spec_rf <- dials::grid_latin_hypercube(
  parameters(model_spec_rf) %>%
    update(mtry = mtry(range = c(1L, 30L)),
           trees = trees(range = c(1L, 3000L))),
  size = 5
)

# Ajuste en paralelo

# Backend
registerDoFuture()

# Cantidad de procesadores 
n_cores <- parallel::detectCores() - 1 # 1 CPU para el S.O.

# Estrategia de ejecucion
plan(
  strategy = cluster,
  workers = parallel::makeCluster(n_cores)
)

# Inicio
tic()

set.seed(123)

tune_rf <- wrkflw_fit_rf %>% 
  update_model(model_spec_rf) %>% 
  tune::tune_grid(
    resamples = resamples_tscv,
    grid = grid_spec_rf,
    metrics = default_forecast_accuracy_metric_set(),
    control = control_grid(verbose = FALSE, save_pred = TRUE)
  )

toc()

# Reset plan
plan(
  strategy = sequential
)

# Mejor modelo

tune_rf %>%
  tune::show_best(
    metric = "rmse", # https://yardstick.tidymodels.org/articles/metric-types.html
    n = Inf)

tune_rf %>%
  tune::show_best(
    metric = "mape",
    n = Inf) %>% 
  print(n=Inf)

grid_plot <- tune_rf %>% 
  tune::autoplot() +
  geom_smooth(se = FALSE) +
  theme_bw()

ggplotly(grid_plot)

# Intento 2

set.seed(123)

grid_spec_rf_2 <- dials::grid_latin_hypercube(
  parameters(model_spec_rf) %>%
    update(mtry = mtry(range = c(10L, 20L)),
           trees = trees(range = c(800L, 1200L))),
  size = 10
)


registerDoFuture()
n_cores <- parallel::detectCores() - 1 

plan(
  strategy = cluster,
  workers = parallel::makeCluster(n_cores)
)

tic()

set.seed(123)

tune_rf_2 <- wrkflw_fit_rf %>% 
  update_model(model_spec_rf) %>% 
  tune_grid(
    resamples = resamples_tscv,
    grid = grid_spec_rf_2,
    metrics = default_forecast_accuracy_metric_set(),
    control = control_grid(verbose = FALSE, save_pred = TRUE)
  )

toc()

# Reset plan
plan(
  strategy = sequential
)

# Mejor modelo

tune_rf_2 %>%
  tune::show_best(
    metric = "rmse", # https://yardstick.tidymodels.org/articles/metric-types.html
    n = Inf)

tune_rf_2 %>%
  tune::show_best(
    metric = "mape",
    n = Inf)

grid_plot_2 <- tune_rf_2 %>% 
  tune::autoplot() +
  geom_smooth(se = FALSE) +
  theme_bw()

ggplotly(grid_plot_2)

# [!] Repetir hasta quedar conforme bajo algun criterio de error

# Reajuste y evaluacion del mejor modelo

set.seed(123)

wrkflw_fit_best_rf <- wrkflw_fit_rf %>% 
  workflows::update_model(model_spec_rf) %>% 
  tune::finalize_workflow(
    tune_rf %>%
      tune::show_best(
        metric = "rmse") %>% 
      dplyr::slice(1)
  ) %>% 
  # Reajustamos con todos los datos de entrenamiento
  fit(training(splits))

best_calibration_tbl <- wrkflw_fit_best_rf %>% 
  modeltime_calibrate(testing(splits))

best_calibration_tbl %>% 
  modeltime_accuracy()

best_calibration_tbl %>% 
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = us_monthly_tbl
  ) %>% 
  plot_modeltime_forecast()

