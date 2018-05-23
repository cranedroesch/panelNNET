
OLStrick_function <- function(parlist, hidden_layers, y, fe_var, lam, parapen, penalize_toplayer, nlayers){
  # parlist <- pnn$parlist
  # hidden_layers <- pnn$hidden_layers
  # y = pnn$y
  # fe_var <- pnn$fe_var
  # lam <- pnn$lam
  # parapen <- pnn$parapen
  # hidden_layers <- hlayers
  Zdm <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
    hidden_layers[[i]][[length(hidden_layers[[i]])]]
  }
  Zdm <- cbind(hidden_layers$param, Zdm)
  if (!is.null(fe_var)){
    Zdm <- demeanlist(Zdm, list(fe_var))
    targ <- demeanlist(y, list(fe_var))      
  }  else {
    targ <- y
  }   
  if (lam >0){
    const <- parlist$beta_param*parapen
    if (penalize_toplayer == TRUE){
      for (i in 1:(length(parlist)-1)){
        const <- c(const, parlist[[i]]$beta)
      }
    }
    constraint <- sum(const^2)
    #set up the penalty vector
    D <- rep(1, ncol(Zdm))
    if(!is.null(parapen)){
      D[1:length(parapen)] <- D[1:length(parapen)]*parapen #incorporate parapen
    }
    if (penalize_toplayer == FALSE){
      D <- D*0
    }
    if (is.null(fe_var)){
      D[1] <- 0
    }
    # scale the data
    starg <- scale(targ)
    sZdm <- colScale(Zdm)
    #deal with cases that don't vary, typically with parametric intercepts
    nancols <- apply(sZdm, 2, function(x){all(is.nan(x))})
    if (any(nancols)){
      sZdm[,nancols] <- 1
    }
    # matmult   
    Zty <- MatMult(t(sZdm), starg)
    ZtZ <- MatMult(t(sZdm), sZdm)
    scalefac <- attr(starg, "scaled:scale")/attr(sZdm, "scaled:scale")
    scalefac[scalefac == Inf] <- 1
    f <- function(L){
      bi <- tryCatch(as.numeric(MatMult(solve(ZtZ + diag(D)*as.numeric(L)), Zty)), error = function(e){b})
      bi <- bi*scalefac
      (crossprod(bi*D) - constraint)^2
    }
    o <- optim(par = lam, f = f, method = 'Brent', lower = lam, upper = 1e9)
    newlam2 <- o$par
    #New top-level params
    b <- tryCatch(as.numeric(MatMult(solve(ZtZ + diag(D)*as.numeric(newlam2)), Zty)), error = function(e){b})
    b <- b*scalefac
    # b <- as.numeric(coef(glmnet(Zdm, targ, lambda = newlam2, standardize = T, intercept = F, alpha = 0, penalty.factor = D)))[-1]
  } else { # when lam is equal to zero
    Zty <- MatMult(t(Zdm), targ)
    ZtZ <- MatMult(t(Zdm), Zdm)
    b <- tryCatch(as.numeric(MatMult(solve(ZtZ), Zty)),
                  error = function(e){b})
  }
  if (inherits(b, "error")){
    print("singularity in OLStrick!")
    return(parlist)
  }    
  if(length(parlist$beta_param)>0){
    parlist$beta_param <- b[1:length(parlist$beta_param)]    
  }
  leftoff <- length(parlist$beta_param)
  for (i in 1:(length(parlist)-1)){
    idx <- (leftoff+1):(leftoff+length(parlist[[i]]$beta))
    parlist[[i]]$beta <- b[idx]
    leftoff <- max(idx)
  }
  return(parlist)
}
