library(modeltime.resample) # TSCV
library(tidymodels)

# Especificacion Regresion Lineal-
model_spec_lm <- linear_reg() %>%
  set_engine("lm")

# model_fit_lm <- model_spec_lm %>%
#   fit(
#     y ~ as.numeric(date) + I(as.numeric(date) ^ 2) + lubridate::month(date, label = TRUE),
#     data = us_monthly_full
#   )

# Especificacion receta
# Forma funcional
recipe_spec <- recipes::recipe(y ~ date, data = us_monthly_full) %>%
  # Agrega Mes como variable categorica
  recipes::step_date(date, features = "month", ordinal = FALSE) %>%
  # Agrega tendencia lineal
  recipes::step_mutate(trend = as.numeric(date)) %>%
  # Agrega tendencia polinomial
  recipes::step_mutate(poly_trend = I(as.numeric(date)) ^ 2) %>%
  #recipes::step_normalize(date_num) %>%
  # Remueve la fecha para ajuste
  recipes::step_rm(date)

# Revisamos el eventual resultado de preprocesamiento
recipe_spec %>%
  prep() %>%
  juice() %>%
  tail(20)

# Se establece un workflow que incorpora la receta de preprocesamiento y la
# la especificacion del modelo a ajustar. El workflow se ajusta para
# establecer la forma funcional
wflw_fit_lm <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_lm) %>%
  fit(us_monthly_full)

# Se extienden los datos originales con registros a ser pronosticados
us_monthly_full <- us_monthly %>%
  timetk::tk_tbl() %>%
  # Horizonte de pronostico h = 12
  timetk::future_frame(.date_var = date,
                       .length_out = "12 months",
                       .bind_data = TRUE)

# Revision de datos a pronosticar en el mismo objeto
us_monthly_full %>%
  tail(15)

# Datos de entrenamiento
us_monthly_tbl <- us_monthly_full %>%
  filter(!is.na(y))

# Datos a pronosticar (fechas)
us_monthly_future_tbl <- us_monthly_full %>%
  filter(is.na(y))

# Cantidad de meses y anios para entrenar y validar
us_monthly_tbl %>%
  summarise(total_months = n_distinct(date)) %>%
  mutate(total_years = total_months / 12)

# Esquema de validacion cruzada
us_monthly_tscv <- us_monthly_tbl %>%
  timetk::time_series_cv(
    date_var    = date,
    initial     = 12 * 15,
    # 15 anios
    assess      = "12 months",
    # Evaluacion
    skip        = "12 months",
    # Desplazamiento de ventana rodante
    cumulative  = TRUE # Acumulacion de datos historicos
  )

# Distancia temporal ascendente en particiones (slices)
# Grafico de esquema de validacion cruzada
us_monthly_tscv %>%
  timetk::tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, y,
                           .facet_ncol = 2, .interactive = TRUE)

# Tabla de modelos (Workflow modeltime)
(models_tbl <- modeltime_table(wflw_fit_lm))

# Ajuste de modelo a distintas particiones
resamples_fitted <- models_tbl %>%
  modeltime.resample::modeltime_fit_resamples(resamples = us_monthly_tscv,
                                              control   = tune::control_resamples(verbose = TRUE))

# Desempenio de modelos a traves de particiones (TSCV)
resamples_fitted %>%
  plot_modeltime_resamples(
    .point_size  = 3,
    .point_alpha = 0.8,
    .interactive = FALSE
  )

# Evaluacion de capacidad predictiva promedio
resamples_fitted %>%
  modeltime_resample_accuracy(summary_fns = mean) %>%
  table_modeltime_accuracy(.interactive = FALSE)

# Pronostico Real
resamples_fitted %>%
  modeltime_forecast(new_data = us_monthly_future_tbl,
                     actual_data = us_monthly_full) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# Grafico comparativo particiones 
# Try-hard: podria ser un feature request para el paquete
slice_plot <- resamples_fitted$.resample_results %>%
  pluck(1) %>%
  mutate(
    training_tbl = map(splits, training),
    testing_tbl = map(splits, testing)
  ) %>%
  select(id, .predictions, training_tbl, testing_tbl) %>%
  mutate(assessment_tbl = pmap(list(training_tbl, testing_tbl, .predictions),
                               function(training_tbl, testing_tbl, .predictions) {
                                 testing_tbl %>%
                                   dplyr::bind_cols(.predictions %>% select(.pred)) %>%
                                   dplyr::bind_rows(training_tbl) %>%
                                   dplyr::mutate(.type = ifelse(is.na(.pred), "Training", "Testing")) %>%
                                   dplyr::arrange(date)
                                 
                               })) %>%
  select(id, assessment_tbl) %>%
  mutate(plot = map(assessment_tbl, function(x) {
    ggplot(data = x, aes(x = date, y = y, color = .type)) +
      geom_line() +
      geom_line(aes(x = date, y = .pred), color = "orange") +
      scale_color_manual(
        name = "Series",
        values = c(
          "Training" = "black",
          "Testing" = "lightblue",
          "Forecast" = "orange"
        )
      )
  }))

library(gridExtra)

grid.arrange(
  slice_plot$plot[[1]],
  slice_plot$plot[[2]],
  slice_plot$plot[[3]],
  slice_plot$plot[[4]],
  slice_plot$plot[[5]],
  ncol = 2
)
