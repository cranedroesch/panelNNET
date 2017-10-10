\name{cv.panelNNET}
\alias{cv.panelNNET}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Cross-validate a panelNNET object
}
\description{
Estimates generalization error on a panelNNET object by k-fold cross-validation on the top layer of the model
}
\usage{
cv.panelNNET(obj, folds = NULL, nfolds = 10, seed = 1, parallel = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{
A fitted panelNNET object
}
  \item{folds}{
An integer vector of pre-specified folds, or NULL
}
  \item{nfolds}{
The number of folds, if folds are to be randomly assigned
}
  \item{seed}{
A random seed
}
  \item{parallel}{
Implement in parallel, or not
}
}
\details{
The top layer of a neural network is a linear model.  This functon estimates generalization error on a panelNNET object, treating it as if it were an OLS/fixed effects regression by ignoring the lower layers.  Note that cross-validation could be done by a full re-fitting, but that would be much, much slower.
}
\value{
A list of the mean and SD of mse values when tested on the k withheld samples.
}
\author{
Andrew Crane-Droesch 
}
\note{
Active development
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
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

#Fitting a two-layer neural net with 5 and 3 hidden units
pnn1 <- panelNNET(y, X, hidden_units = c(3,2)
  , fe_var = id, lam = 1
  , time_var = time, param = P, verbose = FALSE
  , gravity = 1.01
  , RMSprop = TRUE, convtol = 1e-5, maxit = 10000
  , activation = 'tanh', parapen = c(0,0)
)

pnn2 <- panelNNET(y, X, hidden_units = c(4,3)
  , fe_var = id, lam = 1
  , time_var = time, param = P, verbose = FALSE
  , bias_hlayers = TRUE, gravity = 1.01
  , RMSprop = TRUE, convtol = 1e-5, maxit = 10000
  , activation = 'tanh', doscale = TRUE, parapen = c(0,0)
)

pnn1$mse
pnn2$mse

library(parallel)
library(doParallel)
registerDoParallel(detectCores())
cv.panelNNET(pnn1)
cv.panelNNET(pnn2)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
