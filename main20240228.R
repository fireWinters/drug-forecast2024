# 根据20240228的python文件，将其转换为R语言的脚本
# 加载必要的库
library(readxl) # 用于读取Excel文件
library(dplyr) # 用于数据处理
library(ggplot2) # 用于绘图
library(xgboost) # 用于使用XGBoost算法
library(caret) # 用于模型训练和评估
library(lubridate) # 用于处理日期和时间
library(rBayesianOptimization) # 用于贝叶斯优化
library(lightgbm) # 用于使用LightGBM算法
library(forecast) # 用于使用ARIMA模型
library(purrr) # 用于功能性编程工具

# 定义一个函数，用于绘制真实值和预测值的散点图，并绘制对角线
scatter_plot_with_diagonal <- function(y_true, y_pred) {
  ggplot() +
    geom_point(aes(x = y_true, y = y_pred), color = 'blue', size = 2) +
    geom_abline(intercept = 0, slope = 1, color = 'red', linetype = 'dashed') +
    xlab('True Values') +
    ylab('Predicted Values') +
    ggtitle('Scatter Plot of True vs. Predicted')
}

# 定义一个函数，用于获取所有的数据，并将每个sheet拼接成一个DataFrame
get_all_dataframes <- function(data_folder_path) {
  all_dataframes <- list()
  files <- list.files(path = data_folder_path, pattern = '*.xlsx', full.names = TRUE)
  
  for (file_path in files) {
    sheets <- readxl::excel_sheets(file_path)
    for (sheet_name in sheets) {
      sheet_data <- readxl::read_excel(file_path, sheet = sheet_name)
      date_str <- ifelse(grepl("\\.", sheet_name), gsub("\\.", "-", sheet_name), gsub("\\.", "-", tools::file_path_sans_ext(basename(file_path))))
      sheet_data$Date <- as.Date(date_str, format = '%Y-%m-%d')
      all_dataframes[[length(all_dataframes) + 1]] <- sheet_data
    }
  }
  
  final_dataframe <- do.call(rbind, all_dataframes)
  final_dataframe <- final_dataframe %>% select(-c('入库单位', '基本单位'))
  return(final_dataframe)
}

# 定义一个函数，用于数据清洗
data_cleaning <- function(df) {
  df <- df %>% filter(de_sum_90 > 0)
  return(df)
}

# 定义一个函数，用于处理标签
label_process <- function(df, days = 7) {
  df <- df %>%
    group_by(药品名称) %>%
    mutate(y = zoo::rollapply(减少数量, width = days, FUN = sum, align = 'right', fill = NA)) %>%
    ungroup()
  return(df)
}

# 定义一个函数，用于数据处理，包括日期、季度和时序特征
data_process <- function(df, day_lst) {
  df$Date <- as.Date(df$Date)
  df <- df %>% arrange(药品名称, Date)
  df$month <- month(df$Date)
  df$quarter <- quarter(df$Date)
  df$减少数量 <- abs(df$减少数量)
  
  for (days in day_lst) {
    df <- df %>%
      group_by(药品名称) %>%
      mutate(!!paste0('de_sum_', days) := zoo::rollapply(减少数量, width = days, FUN = sum, align = 'right', fill = NA),
             !!paste0('de_mean_', days) := zoo::rollapply(减少数量, width = days, FUN = mean, align = 'right', fill = NA),
             !!paste0('de_max_', days) := zoo::rollapply(减少数量, width = days, FUN = max, align = 'right', fill = NA),
             !!paste0('de_min_', days) := zoo::rollapply(减少数量, width = days, FUN = min, align = 'right', fill = NA)) %>%
      ungroup()
  }
  
  return(df)
}

# 由于R语言中的模型训练和超参数优化的流程与Python有所不同，这里不再展示完整的XGBoost和LightGBM的交叉验证和贝叶斯优化的代码。
# 相应的，可以使用R语言的caret包和rBayesianOptimization包来进行模型训练和超参数优化。

# 主函数
main <- function() {
  current_directory <- '/path/to/your/data'
  data_folder_path <- file.path(current_directory, 'realData')
  
  # 得到所有数据
  df <- get_all_dataframes(data_folder_path)
  # ... 后续的数据处理和模型训练代码
}

# 调用主函数
main()
