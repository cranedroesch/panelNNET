

# defines the 'not in' function
"%ni%" <- Negate("%in%")

# uses eigen if the matrix isn't sparse
MatMult <- function(A, B){
  if ("dgCMatrix" %ni% c(unlist(class(A)), unlist(class(B)))){
    if ("dgeMatrix" %in% c(unlist(class(A)))){
      A <- as.matrix(A)
    }
    if ("dgeMatrix" %in% c(unlist(class(B)))){
      B <- as.matrix(B)
    }
    eigenMapMatMult(A, B)
  } else {
    A %*% B
  }
}

recursive_add <- function(x, y) tryCatch(x + y, error = function(e) Map(recursive_add, x, y))
recursive_subtract <- function(x, y) tryCatch(x - y, error = function(e) Map(recursive_subtract, x, y))
recursive_mult <- function(x, y) tryCatch(x * y, error = function(e) Map(recursive_mult, x, y))

colScale = function(x,
                    center = TRUE,
                    scale = TRUE,
                    add_attr = TRUE,
                    rows = NULL,
                    cols = NULL) {
  
  if (!is.null(rows) && !is.null(cols)) {
    x <- x[rows, cols, drop = FALSE]
  } else if (!is.null(rows)) {
    x <- x[rows, , drop = FALSE]
  } else if (!is.null(cols)) {
    x <- x[, cols, drop = FALSE]
  }
  
  ################
  # Get the column means
  ################
  cm = colMeans(x, na.rm = TRUE)
  ################
  # Get the column sd
  ################
  if (scale) {
    csd = colSds(x, center = cm)
  } else {
    # just divide by 1 if not
    csd = rep(1, length = length(cm))
  }
  if (!center) {
    # just subtract 0
    cm = rep(0, length = length(cm))
  }
  x = t( (t(x) - cm) / csd )
  if (add_attr) {
    if (center) {
      attr(x, "scaled:center") <- cm
    }
    if (scale) {
      attr(x, "scaled:scale") <- csd
    }
  }
  return(x)
}