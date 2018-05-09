

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