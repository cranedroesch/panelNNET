\name{predict.panelNNET}
\alias{predict.panelNNET}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Prediction function for panelNNET objects
}
\description{
Prediction function for panel neural networks
}
\usage{
\method{predict}{panelNNET}(obj, newX = NULL, fe.newX = NULL, new.param = NULL, se.fit = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{
The panelNNET object to be predicted from
}
  \item{newX}{
New X's.  If empty, return the in-sample prediction.
}
  \item{fe.newX}{
A factor of cross-sectional units.  Cross-sectional units must be a subset of those supplied at fitting.
}
  \item{new.param}{
New parametric variables corresponding to the new X's.
}
  \item{se.fit}{
If TRUE, calculate standard errors of the fitted values.  This involves computing the Jacobian for the fitted values, and can be slow.  Currently only numerical Jacobian computation is implemented.  Pointwise standard errors are computed to each paramater covariance matrix stored in the fitted panelNNET object.
}

}
\details{
Prediction function for panelNNET
}
\value{
The predicted values.  If se.fit = TRUE, a matrix with predicted values in the first column, and pointwise standard errors corresponding to each covariance matrix in the object in the subsequent columns.
}
\references{
Friedman, Jerome, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning. Vol. 1. Springer, Berlin: Springer series in statistics, 2001.
}
\author{
Andrew Crane-Droesch
}
\examples{
  set.seed(1)
  #Fake dataset
  N <- 1000
  p <- 20
  X <- as.data.frame(mvrnorm(N, rep(0, p), diag(rep(1, p))))
  id <- factor(0:(N-1)\%\%20+1)
  id.eff <- rnorm(nlevels(id), sd = 5)
  time <- 0:(N - 1)\%/\%20+1
  u <- rnorm(N, sd = 5)
  y <- sin(3*X$V1) - cos(4*X$V2) + 3*tanh((-2*X$V1+X$V2+X$V4)*X$V3) + X$V6/(X$V7+8) + id.eff[id] +
     .5*time - .005*time^2 + u
  hist(y)


  #Parametric and nonparametric terms
  X <- X
  P <- cbind(time, time^2)

  #Traiing and test set
  tr <- time<35
  te <- tr == FALSE

  #Fitting a two-layer neural net with 5 and 3 hidden units
  pnn <- panelNNET(y[tr], X[tr,], hidden_units = c(5,3)
    , fe_var = id[tr], lam = 1
    , time_var = time[tr], param = P[tr,], verbose = FALSE
    , bias_hlayers = TRUE, gravity = 1.01
    , RMSprop = TRUE, convtol = 1e-5, maxit = 10000
    , activation = 'tanh', doscale = TRUE, parapen = c(0,0)
  )

  plot(pnn)
  summary(pnn)


  ##Predicting for the test set
  pr <- predict(pnn, newX = as.matrix(X[te,]), fe.newX = id[te], new.param = P[te,])
  plot(y[te], pr)
  mean((y[te] - pr)^2)
  abline(0,1)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
