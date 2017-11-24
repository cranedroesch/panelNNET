

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
