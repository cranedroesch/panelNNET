

OLStrick_function <- function(parlist, hidden_layers, y, fe_var, lam, parapen){
# parlist <- pnn$parlist
# hidden_layers <- pnn$hidden_layers
# y = pnn$y
# fe_var <- pnn$fe_var
# lam <- pnn$lam
# parapen <- pnn$parapen
# hidden_layers <- hlayers
  constraint <- sum(c(parlist$beta_param*parapen, parlist$beta)^2)
  #getting implicit regressors depending on whether regression is panel
  if (!is.null(fe_var)){
    Zdm <- demeanlist(as.matrix(hidden_layers[[length(hidden_layers)]]), list(fe_var))
    targ <- demeanlist(y, list(fe_var))
  } else {
    Zdm <- hidden_layers[[length(hidden_layers)]]
    targ <- y
  }
  #set up the penalty vector
  D <- rep(1, ncol(Zdm))
  if (is.null(fe_var)){
    pp <- c(0, parapen) #never penalize the intercept
  } else {
    pp <- parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  # find implicit lambda
  b <- c(parlist$beta_param, parlist$beta)
  newlam <-   1/constraint * MatMult(t(b), (MatMult(t(Zdm), targ) - MatMult(MatMult(t(Zdm), Zdm),b)))
  newlam <- max(lam, newlam) #dealing with the case where you're not constrained
  #New top-level params
  
  
  OLStrick_function <- function(parlist, hidden_layers, y, fe_var, lam, parapen){
    # parlist <- pnn$parlist
    # hidden_layers <- pnn$hidden_layers
    # y = pnn$y
    # fe_var <- pnn$fe_var
    # lam <- pnn$lam
    # parapen <- pnn$parapen
    # hidden_layers <- hlayers
    constraint <- sum(c(parlist$beta_param*parapen, parlist$beta)^2)
    #getting implicit regressors depending on whether regression is panel
    if (!is.null(fe_var)){
      Zdm <- demeanlist(as.matrix(hidden_layers[[length(hidden_layers)]]), list(fe_var))
      targ <- demeanlist(y, list(fe_var))
    } else {
      Zdm <- hidden_layers[[length(hidden_layers)]]
      targ <- y
    }
    #set up the penalty vector
    D <- rep(1, ncol(Zdm))
    if (is.null(fe_var)){
      pp <- c(0, parapen) #never penalize the intercept
    } else {
      pp <- parapen #parapen
    }
    D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
    # find implicit lambda
    b <- c(parlist$beta_param, parlist$beta)
    newlam <-   1/constraint * MatMult(t(b), (MatMult(t(Zdm), targ) - MatMult(MatMult(t(Zdm), Zdm),b)))
    newlam <- max(lam, newlam) #dealing with the case where you're not constrained
    #New top-level params
    DBG <<- list(Zdm = Zdm,
                 D = D,
                 newlam = newlam,
                 targ = targ)
    b <- as.numeric(MatMult(MatMult(solve(MatMult(t(Zdm),Zdm) + diag(D)*as.numeric(newlam)), t(Zdm)), targ))
    parlist$beta_param <- b[1:length(parlist$beta_param)]
    parlist$beta <- b[(length(parlist$beta_param)+1):length(b)]
    return(parlist)
  }
  
  
  OLStrick_function_old <- function(parlist, hidden_layers, y, fe_var, lam, parapen){
    # parlist <- pnn$parlist
    # hidden_layers <- pnn$hidden_layers
    # y = pnn$y
    # fe_var <- pnn$fe_var
    # lam <- pnn$lam
    # parapen <- pnn$parapen
    # hidden_layers <- hlayers
    constraint <- sum(c(parlist$beta_param*parapen, parlist$beta)^2)
    #getting implicit regressors depending on whether regression is panel
    if (!is.null(fe_var)){
      Zdm <- demeanlist(as.matrix(hidden_layers[[length(hidden_layers)]]), list(fe_var))
      targ <- demeanlist(y, list(fe_var))
    } else {
      Zdm <- hidden_layers[[length(hidden_layers)]]
      targ <- y
    }
    #set up the penalty vector
    D <- rep(1, ncol(Zdm))
    if (is.null(fe_var)){
      pp <- c(0, parapen) #never penalize the intercept
    } else {
      pp <- parapen #parapen
    }
    D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
    #function to find implicit lambda
    f <- function(lam){
      bi <- eigenMapMatMult(solve(eigenMapMatMult(t(Zdm),Zdm) + diag(D)*lam), t(Zdm)) %*% targ
      (t(bi*D) %*% (bi*D) - constraint)^2
    }
    #optimize it
    o <- optim(par = lam, f = f, method = 'Brent', lower = lam, upper = 1e9)
    #new lambda
    newlam <- o$par
    b <- as.numeric(MatMult(MatMult(solve(MatMult(t(Zdm),Zdm) + diag(D)*as.numeric(newlam)), t(Zdm)), targ))
    parlist$beta_param <- b[1:length(parlist$beta_param)]
    parlist$beta <- b[(length(parlist$beta_param)+1):length(b)]
    return(parlist)
  }
  
  
  
  b <- as.numeric(MatMult(MatMult(solve(MatMult(t(Zdm),Zdm) + diag(D)*as.numeric(newlam)), t(Zdm)), targ))
  parlist$beta_param <- b[1:length(parlist$beta_param)]
  parlist$beta <- b[(length(parlist$beta_param)+1):length(b)]
  return(parlist)
}


OLStrick_function_old <- function(parlist, hidden_layers, y, fe_var, lam, parapen){
  # parlist <- pnn$parlist
  # hidden_layers <- pnn$hidden_layers
  # y = pnn$y
  # fe_var <- pnn$fe_var
  # lam <- pnn$lam
  # parapen <- pnn$parapen
  # hidden_layers <- hlayers
  constraint <- sum(c(parlist$beta_param*parapen, parlist$beta)^2)
  #getting implicit regressors depending on whether regression is panel
  if (!is.null(fe_var)){
    Zdm <- demeanlist(as.matrix(hidden_layers[[length(hidden_layers)]]), list(fe_var))
    targ <- demeanlist(y, list(fe_var))
  } else {
    Zdm <- hidden_layers[[length(hidden_layers)]]
    targ <- y
  }
  #set up the penalty vector
  D <- rep(1, ncol(Zdm))
  if (is.null(fe_var)){
    pp <- c(0, parapen) #never penalize the intercept
  } else {
    pp <- parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  #function to find implicit lambda
  f <- function(lam){
    bi <- eigenMapMatMult(solve(eigenMapMatMult(t(Zdm),Zdm) + diag(D)*lam), t(Zdm)) %*% targ
    (t(bi*D) %*% (bi*D) - constraint)^2
  }
  #optimize it
  o <- optim(par = lam, f = f, method = 'Brent', lower = lam, upper = 1e9)
  #new lambda
  newlam <- o$par
  b <- as.numeric(MatMult(MatMult(solve(MatMult(t(Zdm),Zdm) + diag(D)*as.numeric(newlam)), t(Zdm)), targ))
  parlist$beta_param <- b[1:length(parlist$beta_param)]
  parlist$beta <- b[(length(parlist$beta_param)+1):length(b)]
  return(parlist)
}


