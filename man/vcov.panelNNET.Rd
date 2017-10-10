\name{vcov.panelNNET}
\alias{vcov.panelNNET}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Approximate variance-covariance matrix of a panelNNET object
}
\description{
Calculates the approximate variance covariance of either the linear taylor expansion or the OLS approximation of the panelNNET object.  For each of these, calculates the covariance matrix assuming homoskedasticity, the "sandwich" estimator, or the "cluster-sandwich" estimator.
}
\usage{
vcov.panelNNET(obj, option, J = NULL, numeric = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{obj}{
The fitted panelNNET object
}
  \item{option}{
Which sort of covariance matrix you want.  Options are "Jacobian_homoskedastic", "Jacobian_sandwich", "Jacobian_cluster", "OLS", "sandwich", and "cluster"
}
  \item{J}{
A supplied Jacobian matrix
}
  \item{numeric}{
Whether the Jacobian should be computed through numerical approximation.  Defaults to TRUE.
}
}
\details{
All of these covariance matrices are approximations, and inference based on them should be used very cautiously.
}
\value{
  \item{vc }{The covariance matrix approximation/estimate}
  \item{Jacobian }{The Jacobian}
}
%\references{
%%% ~put references to the literature/web site here ~
%}
\author{
Andrew Crane-Droesch
}
\note{
This currently doesn't work for single-layer networks.
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
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
  , gravity = 1.01
  , RMSprop = TRUE, convtol = 1e-5, maxit = 10000
  , activation = 'tanh', parapen = c(0,0)
)

vc <- vcov(pnn, 'Jacobian_cluster')
lapply(vc, dim)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
