dosci <-
function(x,l=4){
#x = .00359
  if(abs(log10(abs(x)))>l){
    x <- format(x, scientific = TRUE, digits = l)
  }
  x
}
