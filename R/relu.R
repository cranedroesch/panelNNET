relu <-
function(v){
  v[v<0] <- 0
  v
}

relu_prime <- 
function(v){
  v[v<0] <- 0
  v[v>=0] <- 1
  v
}

lrelu <-
function(v){
  v[v<0] <- v[v<0]*.01
  v
}

lrelu_prime <- 
function(v){
  v[v>=0] <- 1
  v[v<0] <- 0.01
  v
}


