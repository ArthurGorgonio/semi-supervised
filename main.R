library(rminer)
library(RWeka)

#' @description This function set the class feature to NA without change the
#'  class of selected samples
#'
#' @usage new_base(labeled, train_id)
#'
#' @param labeledDB The full data set without changes
#' @param trainId The vector with the selected samples
#'
#' @return A new data set with some percents of the samples have the NA in class
#' feature
#'
new_base <- function(labeled, train_id){
  labeled[-train_id, "class"] <- NA
  return(labeled)
}

calculate_centroid <- function(labeled) {
  classes <- droplevels(labeled$class)
  features <- ncol(labeled) - 1
  centroid <- matrix(rep(0, features), nrow = length(classes), ncol = features)
  rownames(centroid) <- classes
  for(cl in classes) {
    instances <- which(labeled$class == cl)
    for (feature in 1:features) {
      centroid[cl, feature] <- mean(labeled[instances, feature])
    }
  }
  return (centroid)
}

calculate_metric <- function(instance, centroids) {
  metrics <- c()
  for (cent in 1:nrow(centroids)) {
    metrics <- union(metrics, euclidian_distance(instance, centroids[cent,]))
  }
  return (metrics)
}

calculate_threshold <- function(labeled, centroids) {
  best_dist <- c()
  for (instance in 1:nrow(labeled)) {
    metrics <- calculate_metric(labeled[instance,], centroids)
    best_dist <- c(best_dist, min(metrics))
  }
  return (max(best_dist) * 1.1)
}

euclidian_distance <- function(a, b) {
  return (sqrt(sum((a - b)^2)))
}

pre_processing <- function(data) {
  for (i in 1:(ncol(data)-1)) {
    data[,i] <- (data[,i] - min(data[,i])) / (max(data[,i]) - min(data[,i]))
  }
  return (data)
}

select_instances <- function(unlabeled, centroids, threshold) {
  selected <- list()
  for (instance in 1:nrow(unlabeled)) {
    metrics <- calculate_metric(unlabeled[instance,], centroids)
    if (min(metrics) < threshold) {
      selected$id <- union(selected$id, as.numeric(rownames(unlabeled[instance,])))
      selected$label <- c(selected$label, rownames(centroids)[which.min(metrics)])
    }
  }
  return (selected)
}

self_training <- function(data, sup, correct_label, max_its = 100) {
  it <- 0
  total_intances <- nrow(data)
  class_pos <- match("class", colnames(data))
  centroids <- calculate_centroid(data[sup,])
  threshold <- calculate_threshold(data[sup, -class_pos], centroids)
  threshold_old <- -1
  while ((it < max_its) && (length(sup) < total_intances) &&
         (threshold_old != threshold)) {
    it <- it + 1
    selected <- select_instances(data[-sup, -class_pos], centroids, threshold)
    if (length(selected$label) > 0) {
      data$class[selected$id] <- selected$label
      sup <- union(sup, selected$id)
      centroids <- calculate_centroid(data[sup,])
      cat("\nIteration INFO!\n\n")
      cat("IT: ",it, '\n')
      cat(sum(correct_label[selected$id] == data$class[selected$id]),
          "/", length(selected$label))
    } else {
      threshold_old <- threshold
      threshold <- calculate_threshold(data[sup, -class_pos], centroids)
    }
  }
  return (data$class)
}





source("src/statistics.R")
source("src/database.R")
source("src/crossValidation.R")
source("src/write.R")

datasets <- list.files("datasets/")

seeds <- c(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113)

for (dataset in datasets) {
  data_name <- strsplit(dataset, ".", T)[[1]][1]
  cat("\n\n",data_name,"\n\n")
  acc_test <- c()
  fmeasure_test <- c()
  precision_test <- c()
  recall_test <- c()
  data <- get_database(dataset, "datasets")
  begin <- Sys.time()
  for (seed in seeds) {
    ids_label <- holdout(data$class, 0.1, mode = "random", seed = seed)
    sup <- as.numeric(rownames(data)[ids_label$tr])
    data2 <- new_base(data, sup)
    predicted_labels <- self_training(data2, sup, data$class)
    cm <- confusion_matrix(predicted_labels, data$class)
    acc_test <- c(acc_test, getAcc(cm))
    fmeasure_test <- c(fmeasure_test, fmeasure(cm))
    precision_test <- c(precision_test, precision(cm))
    recall_test <- c(recall_test, recall(cm))
    cat("\n\nMatriz de Confusão!!\n\n")
    print(cm)
    cat("\nMétricas\n\nACC:       ", round(getAcc(cm), 4) * 100,
        "%\nF-Score:   ", round(fmeasure(cm), 4) * 100,
        "%\nPrecision: ", round(precision(cm), 4) * 100,
        "%\nRecall:    ", round(recall(cm), 4) * 100, "%\n")
  }
  end <- Sys.time()
  write_archive("SimpleEuclidian.txt", ".", data_name, "Euclidian Distance",
               acc_test, fmeasure_test, precision_test, recall_test, begin, end)
}


