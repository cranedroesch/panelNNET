

# this function generates a hinton plot from a convolutional net.  
# it still needs work -- axis labels and formatting, generally.
# it will likely break as soon as 3D nets are implemented, because of how it defines the second row of the plot

hintonPlot <- function(pnn, varnames = NULL, secondrow = TRUE){
  # extract the convolutional parlist
  pp <- pnn$parlist[[1]]  
  # lose the redundant info
  densemat <- apply(pp[,1:(pnn$convolutional$N_TV_layers*pnn$convolutional$Nconv)], 2, function(x){
    x[x!=0]
  })
  tdm <- t(densemat)
  pars <- t(tdm[!duplicated(tdm),])[-1,]
  #set up the plot region
  
  par(mfrow = c(1+secondrow, pnn$convolutional$Nconv))
  for (i in 1:pnn$convolutional$Nconv){
    mat <- matrix(pars[,i], nrow = pnn$convolutional$span*2+1)
    posmat <- negmat <- mat
    posmat[posmat<0] <- 0
    negmat[negmat>0] <- 0
    negmat <- negmat*-1
    image(x = 1:nrow(mat), y = 1: ncol(mat), z = mat,
          xlab = "", ylab = "", xaxt = "n", yaxt = "n",
          col = rgb(negmat/max(abs(mat)), posmat/max(abs(mat)),0)
    )
  }
  if(secondrow == TRUE){
    outgo <- pnn$parlist[[2]][2:(pnn$convolutional$N_TV_layers*pnn$convolutional$Nconv+1),]
    outgo <- apply(outgo, 1, function(x){sqrt(sum(x^2))})
    outgo <- matrix(outgo, ncol = pnn$convolutional$Nconv)
    for (i in 1:pnn$convolutional$Nconv){
      plot(outgo[,i], type = "l", xlab = "", ylab = "", xaxt = "n", yaxt = "n")
    }
  }
}
