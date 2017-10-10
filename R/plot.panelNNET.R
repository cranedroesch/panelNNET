plot.panelNNET <-
function(x, y = NULL, logmse = FALSE,...){
  par(mfrow = c(1,2))
  plot(x$y, x$yhat)
  if(logmse){
    plot(log10(x$msevec))
  } else {
    plot(x$msevec)
  }
}
