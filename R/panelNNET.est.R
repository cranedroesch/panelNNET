panelNNET.est <- function(y, X, hidden_units, fe_var, maxit, lam, time_var, param,
                          parapen, penalize_toplayer, parlist, verbose,
                          report_interval, gravity, convtol, RMSprop, start_LR,
                          activation, batchsize, maxstopcounter, OLStrick, OLStrick_interval,
                          initialization, dropout_hidden, dropout_input, convolutional,
                          LR_slowing_rate, return_best, stop_early, ...){

#   y = dat$dtrat[is]
#   X = Xtr
#   hidden_units = list(c(10,1), c(10,1), c(10,1), c(10,1), c(10,1),c(10,1))
#   fe_var = as.factor(out$cl)
#   maxit = 10
#   lam = 0
#   param = NULL
#   verbose = T
#   report_interval = 1
#   gravity = 1.01
#   convtol = 1e-8
#   activation = 'lrelu'
#   start_LR = .0001
#   parlist = NULL
#   OLStrick = TRUE
#   OLStrick_interval = 1
#   batchsize = length(is)
#   maxstopcounter = 25
#   LR_slowing_rate = 2
#   parapen = NULL
#   return_best = TRUE
#   dropout_hidden = 1
#   dropout_input = 1
# 
# 
# RMSprop = TRUE
# dropout_input <- dropout_hidden <- TRUE
# convolutional <- NULL
# initialization = "HZRS"
# penalize_toplayer = FALSE
# stop_early = NULL

  ##########
  #Define internal functions
  recursive_RMSprop <- function(x, y) tryCatch(LR/sqrt(x+1e-10) * y, error = function(e) Map(recursive_RMSprop, x, y))

  makeConvLayer <- function(convParms, convBias){
    # time-varying portion
    TV <- foreach(i = 1:convolutional$Nconv, .combine = cbind) %do% {
      apply(convMask[,1:N_TV_layers], 2, function(x){# this assumes that the feature detectors have identical shapes
        x[x!=0][-1] <- convParms[[i]]
        x[1] <- convBias[i]
        return(x)
      })
    }
    NTV <- convMask[,colnames(convMask) %ni% convolutional$topology]
    return(Matrix(cbind(TV, NTV)))
  }
  
  check_earlystop <- function(stop_early){ #function placed here so it can leach off of parent env.
    if (is.null(fe_var)){
      fe_output = NULL; 
    } else {
      # compute FE output
      Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
        hlayers[[i]][[length(hlayers[[i]])]]
      }
      Z <- cbind(param, as.matrix(Z))
      Zdm <- demeanlist(as.matrix(Z), list(fe_var))
      B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
      fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(parlist$beta_param, B)))
      fe_output <- data.frame(fe_var, fe)
    }
    pr_obj <- list(parlist = parlist,
                   activation = activation,
                   X = X,
                   param = param,
                   yhat = yhat,
                   fe_var = unique(fe_var),
                   fe = fe_output,
                   convolutional = NULL,
                   hidden_layers = hidden_units)
    pr_test <- predict.panelNNET(obj = pr_obj, 
                                 y_test = stop_early$y_test,
                                 newX = stop_early$X_test, 
                                 fe.newX = stop_early$fe_test, 
                                 new.param = stop_early$P_test)
    return(pr_test)
  }
  
  ###########################
  # sanity checks.  here place checks to ensure that arguments supplied will yield sensible output
  ###########################
  if (gravity <= 1){stop("Gravity must be >1")}
  if (start_LR <= 0){stop("Learning rate must be positive")}
  # if (LR_slowing_rate <= 1){stop("LR_slowing_rate must larger than 1")}
  ###########################
  # start fitting
  ###########################
  # do scaling
  if (class(X) == "list"){
    X <- lapply(X, colScale)
  } else { # make it a list if it isn't
    X <- list(colScale(X))
    hidden_units <- list(hidden_units)
  }
  if (!is.null(param)){
    param <- colScale(param)
  }
  if (activation == 'tanh'){
    activ <- tanh
    activ_prime <- tanh_prime
  }
  if (activation == 'logistic'){
    activ <- logistic
    activ_prime <- logistic_prime
  }
  if (activation == 'relu'){
    activ <- relu
    activ_prime <- relu_prime
  }
  if (activation == 'lrelu'){
    activ <- lrelu
    activ_prime <- lrelu_prime
  }
  if (class(hidden_units) == "list"){
    nlayers <- sapply(hidden_units, length)
  } else {
    nlayers <- length(hidden_units)
  }
  # initialize the convolutional layer, if present
  if (!is.null(convolutional)){
    # set set the topology to start at 1, if it isn't already there.  give a warning if it isn't.
    if (min(convolutional$topology, na.rm =T)>1){
      convolutional$topology <- convolutional$topology - min(convolutional$topology, na.rm =T) +1 
      if (verbose == TRUE){
        print("minimum value in supplied topology greater than 1.  subtracting to get it to start at 1.")
        warning("minimum value in supplied topology greater than 1.  subtracting to get it to start at 1.")
      }
    }
    # make the convolutional masking matrix if using conv nets
    convMask <- convolutional$convmask <- makeMask(X, convolutional$topology, convolutional$span, convolutional$step, convolutional$Nconv)
    # store the number of time-varying variables
    # both in the local env for convenience, and in the convolutional object for passing to other functions
    N_TV_layers <- convolutional$N_TV_layers <- sum(unique(colnames(convMask)) %in% convolutional$topology)
    # For each convolutional "column", initialize the single parameter vector that will be shared among columns
    if (is.null(convolutional$convParms)){
      convParms <- convolutional$convParms <- foreach(i = 1:convolutional$Nconv) %do% {
        rnorm(sum(convMask[-1,1]), sd = 2/sqrt(sum(convMask[-1,1])))
      }
    }
    # Initialize convolutional layer bias, if not present
    # new version: bias terms are not individual to each span, but shared by each span
    if (is.null(convolutional$convBias)){
      convBias <- rnorm(convolutional$Nconv, sd = 2/sqrt(sum(convMask[,1])))
    }
    # initialize the convolutional parlist, if not present
    if (is.null(convolutional$convParMat)){
      convParMat <- convolutional$convParMat <- makeConvLayer(convParms, convBias)
    }
  }
  #get starting weights, either randomly or from a specified parlist
  if (is.null(parlist)){#random starting weights
    parlist <- foreach(i = 1:length(nlayers)) %do% {
      vector('list', nlayers[i])
    }
    names(parlist) <- paste0("p", 1:length(parlist))
    for (p in 1:length(nlayers)){
      for (i in 1:nlayers[p]){
        if (i == 1){
          D <- ncol(X[[p]])
        } else {
          D <- hidden_units[[p]][i-1]
        }
        if (initialization %ni% c('XG', 'HZRS')){#random initialization schemes
          ubounds <- .7 #follows ESL recommendaton
        } else {
          if (initialization == 'XG'){
            ubounds <- sqrt(6)/sqrt(D+hidden_units[[p]][i]+2)#2 is for the bias.  Not sure why 2.  Would need to go back and read the paper.  
          }
          if (initialization == 'HZRS'){
            ubounds <- 2*sqrt(6)/sqrt(D+hidden_units[[p]][i]+2)#2 is for the bias.  Not sure why 2.  Would need to go back and read the paper.
          }
        }
        parlist[[p]][[i]] <- matrix(runif((hidden_units[[p]][i])*(D+1), -ubounds, ubounds), ncol = hidden_units[[p]][i])
      }        
      # vector of parameters at the top layer
      parlist[[p]]$beta <- runif(hidden_units[[p]][i], -ubounds, ubounds)
    }
    # parameters on parametric terms
    if (is.null(param)){
      parlist$beta_param <-  NULL
    } else {
      parlist$beta_param <- runif(ncol(param), -ubounds, ubounds)
    }
    #add the bias term/intercept onto the front, if there are no FE's
    parlist$beta_param <- c(runif(is.null(fe_var), -ubounds, ubounds), parlist$beta_param)
  }
  parlist <- as.relistable(parlist)
  #if there are no FE's, append a 0 to the front of the parapen vec, to leave the intercept unpenalized
  if (is.null(param) & is.null(fe_var)){
    parapen <- 0
  }
  if (is.null(param) & !is.null(fe_var)){
    parapen <- NULL
  }
  if (!is.null(param) & is.null(fe_var)){
    parapen <- c(0, parapen)
  } # if both are not null, then the parapen is the parapen
  #compute hidden layers given parlist
  hlayers <- calc_hlayers(parlist, X = X, param = param, 
                          fe_var = fe_var, nlayers = nlayers, 
                          convolutional = convolutional, activation = activation)
  #calculate ydm and put it in global... ...one year later, I forget why this needs to be in global
  if (!is.null(fe_var)){
    ydm <<- demeanlist(y, list(fe_var)) 
  }
  #####################################
  #start setup
  #get starting mse
  yhat <- as.numeric(getYhat(parlist, hlayers, param, y, ydm, fe_var, nlayers))
  # if using early stopping, initialize object for passing to predict function
  if (!is.null(stop_early)){
    pr_test <- check_earlystop(stop_early)
    best_mse <- mse_test_vec <- mse_test <- msetest_old <- mean((stop_early$y_test-pr_test)^2)
  } else {mse_test <- best_mse <- NULL}
  # in-sample MSE and loss
  mse <- mseold <- mean((y-yhat)^2)
  B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
  lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
  loss <- mse + lam * sum(c(parlist$beta_param*parapen,
                            B*as.numeric(penalize_toplayer),
                            lowerpar)^2)
  LRvec <- LR <- start_LR# starting step size
  #Calculate gradients
  grads <- calc_grads(parlist, hlayers, X, y, yhat, droplist = NULL, nlayers, activ_prime = activ_prime)
  # # check gradients
  # eps <- 1e-5
  # l <- function(pl){
  #   hl <- calc_hlayers(pl, X = X, param = param,
  #                           fe_var = fe_var, nlayers = nlayers,
  #                           convolutional = convolutional, activation = activation)
  #   yh <- as.numeric(getYhat(pl, hl, param, y, ydm, fe_var, nlayers))
  #   MSE <- mean((y-yh)^2)
  #   B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
  #   lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
  #   loss <- MSE + lam * sum(c(pl$beta_param*parapen,
  #                             B*as.numeric(penalize_toplayer),
  #                             lowerpar)^2)
  #   return(loss)
  # }
  # gcheck <- function(pos){
  #   pl <- parlist %>% as.relistable %>% unlist
  #   bvec <- rep(0, length(pl))
  #   bvec[pos] <- eps
  #   plup <- (pl + bvec) %>% relist
  #   pldn <- (pl - bvec) %>% relist
  #   gsim <- (l(plup) - l(pldn))/(2*eps)
  #   comp <- (grads %>% as.relistable %>% unlist)[pos]
  #   print(paste0("calculated: ", comp))
  #   print(paste0("computed: ", gsim))
  #   print(paste0("difference: ", comp - gsim))
  #   print(paste0("ratio: ", comp/gsim))
  # 
  # }
  # gcheck(1)

  #Initialize updates
  updates <- relist(unlist(parlist)*0)
  #initialize G2 term for RMSprop
  if (RMSprop == TRUE){G2 <- updates} else {G2 <- NULL}
  # initialize terms used in the while loop
  D <- 1e6
  ES_stopcounter <- stopcounter <- iter <- 0
  msevec <- lossvec <- c()
  lossvec <- append(lossvec, loss)
  msevec <- append(msevec, mse)
  parlist_best <- parlist
  ###############
  #start iterating
  while(iter < maxit & stopcounter < maxstopcounter){
    #Assign batches
    batchid <- sample(1:length(y) %/% batchsize +1)
    if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
      batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)), replace = TRUE)
    }
    for (bat in 1:max(batchid)) { # run minibatch
      iter <- iter + 1
      curBat <- which(batchid == bat)
      # Create droplist.  
      droplist <- gen_droplist(dropout_hidden, dropout_input, nlayers, hlayers, X)
      #Drop parameters.  
      plist <- drop_parlist(parlist, droplist, nlayers)
      #Compute hidden layers based on parameters
      if (is.null(droplist)){
        Xd <- lapply(X, function(X){X[curBat,]})
      } else {
        Xd <- foreach(i = 1:length(X)) %do% {
          X[[i]][curBat,droplist[[i]][[1]]]
        }
      }
      hlbatch <- calc_hlayers(plist, X = Xd,
                              param = param[curBat,], fe_var = fe_var[curBat], 
                              nlayers = nlayers, convolutional = convolutional, activ = activation)

      # Get yhat from that
      yhat <- getYhat(plist, hlay = hlbatch, param[curBat,], y[curBat], ydm[curBat], fe_var[curBat], nlayers) # update yhat for purpose of computing gradients
      # before updating gradients, compute square of gradients for RMSprop
      if (RMSprop ==  TRUE){
        oldG2 <- rapply(grads, function(x){.9*x^2}, how = "list")
      } 
      # Calculate gradients for minibatch
      grads_p <- calc_grads(plist = plist, hlay = hlbatch, Xd = Xd, y = y[curBat]
        , yhat = yhat, droplist = droplist, nlayers = nlayers, activ_prime = activ_prime)
      grads <- reconstitute(grads_p, droplist, parlist, nlayers) # put zeros back in after dropout...
      # grads <- grads_p
      # Calculate updates to parameters based on gradients and learning rates
      if (RMSprop == TRUE){
        newG2 <- rapply(grads, function(x){.1*x^2}, how = "list")
        G2 <- recursive_add(newG2, oldG2)
        updates <- recursive_RMSprop(G2, grads)
        # }
      } else { #if RMSprop == FALSE
        updates <- rapply(grads, function(x){LR*x}, how = "list")
      }
      # weight decay
      if (lam != 0) {
        # map lambda, the parapen, and the learning rate to the updates
        ppmask <- rapply(parlist, function(x){x^0}, how = "list")
        ppmask$beta_param <- ppmask$beta_param*parapen
        if (penalize_toplayer == FALSE){
          for (i in 1:length(nlayers)){
            ppmask[[i]]$beta <- ppmask[[i]]$beta*0
          }
        }
        wd <- rapply(parlist, function(x){x*lam*LR}, how = "list")
        wd <- recursive_mult(wd, ppmask)
        updates <- as.relistable(recursive_add(updates, wd))
        # don't update the pass-through weights for the non-time-varying variables when using conv 
        if (!is.null(convolutional)){
          updates[[1]][,colnames(updates[[1]]) %ni% convolutional$topology] <- 0
        }
      }
      # Update parameters from update list
      # where there is no parametric term (no fe and no params specified), append a zero-length term to the updates list
      if (length(parlist$beta_param)==0){
        parlist$beta_param <- NULL
        parlist <- recursive_subtract(parlist, updates)
        parlist$beta_param <- rep(0,0)
      } else {
        parlist <- recursive_subtract(parlist, updates)
      }
      if (OLStrick == TRUE & (iter %% OLStrick_interval == 0 | iter == 0)){ # do OLStrick on first iteration
        # Update hidden layers
        hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var,
                                nlayers = nlayers, convolutional = convolutional, activ = activation)
        # OLS trick!
        parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
                                     , fe_var = fe_var, lam = lam, parapen = parapen
                                     , penalize_toplayer, nlayers = nlayers)
      }
      #update yhat for purpose of computing loss function. need to update hidden layers in order to compute loss
      hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var,
                              nlayers = nlayers, convolutional = convolutional, activ = activation)
      yhat <- getYhat(parlist, hlay = hlayers, param, y, ydm, fe_var, nlayers) # update yhat for purpose of computing gradients
      mse <- mseold <- mean((y-yhat)^2)
      B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
      lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
      loss <- mse + lam * sum(c(parlist$beta_param*parapen,
                                B*as.numeric(penalize_toplayer),
                                lowerpar)^2)
      oldloss <- lossvec[length(lossvec)]
      oldmse <- msevec[length(msevec)]
      lossvec <- append(lossvec, loss)
      msevec <- append(msevec, mse)
      # depending on whether loss decreases, increase or decrease learning rate
      if (oldloss <= loss){
        LR <- LR/gravity^LR_slowing_rate
        stopcounter <- stopcounter + 1
        if(verbose == TRUE){
          print(paste0("Loss increased.  Stopcounter now at ", stopcounter))
        }
      } else { # if loss doesn't increase
        LR <- LR*gravity      #gravity...
        
        # check for convergence
        D <- oldloss - loss
        if (D < convtol){
          stopcounter <- stopcounter + 1
          if(verbose == TRUE){print(paste('slowing!  Stopcounter now at ', stopcounter))}
        } else { # reset stopcounter if not slowing per convergence tolerance
          stopcounter <- 0
          # check and see if loss has been up for a while
          bestloss <- which.min(lossvec)
          if(length(lossvec) - bestloss > maxstopcounter*2){
            if(verbose == TRUE){
              print("loss been above minimum for > 2*maxstopcounter")
            }
            stopcounter <- maxstopcounter+1
          }
        }
      }   
      # check to see if early stopping is warranted
      if (!is.null(stop_early)){
        if (iter %% stop_early$check_every == 0 | iter == 0){
          pr_test <- check_earlystop(stop_early)
          mse_test <- mean((stop_early$y_test-pr_test)^2)
          mse_test_vec <- append(mse_test_vec, mse_test)
          if (mse_test == min(mse_test_vec)){
            parlist_best <- parlist
            ES_stopcounter <- 0
            best_mse <- mse_test
            if (verbose == TRUE){
              print(paste0("new low in test set: ", mse_test))
            }
          } else {
            ES_stopcounter <- ES_stopcounter + 1
            if (ES_stopcounter > stop_early$max_ES_stopcounter){
              if(verbose == TRUE){
                print(paste0("test set MSE not improving after ", stop_early$max_ES_stopcounter, " checks"))
              }
              stopcounter <- maxstopcounter+1
            }
          }
        }
      } else { # if not doing early stopping, the best parlist is the one that attains the minimum loss function
        # if achieving a new minimum, stash parlist in parlist_best
        if (loss == min(lossvec)){
          parlist_best <- parlist
        }
      }
      LRvec[iter+1] <- LR
      # verbosity
      if  (verbose == TRUE & iter %% report_interval == 0){
        writeLines(paste0(
          "*******************************************\n"
          , 'Lambda = ',lam, "\n"
          , "Hidden units -> ",paste(hidden_units, collapse = ' '), "\n"
          , " Batch size is ", batchsize, " \n"
          , " Completed ", iter %/% max(batchid), " epochs. \n"
          , " Completed ", bat, " batches in current epoch. \n"
          , "mse is ",mse, "\n"
          , "last mse was ", oldmse, "\n"
          , "difference is ", oldmse - mse, "\n"
          , "loss is ",loss, "\n"
          , "last loss was ", oldloss, "\n"
          , "difference is ", oldloss - loss, "\n"
          , "input layer dropout probability: ", dropout_input, "\n"
          , "hidden layer dropout probability: ", dropout_hidden, "\n"
          , "*******************************************\n"  
        ))
        if (iter>1){
          if (is.null(stop_early)){
            par(mfrow = c(3,2))
          } else {
            par(mfrow = c(4,2))
          }
          if(length(y)>5000){
            plot(1, cex = 0, main = "more than 5000 obs -- not plotting scatter")
          } else {
            plot(y, yhat, col = rgb(1,0,0,.5), pch = 19, main = 'in-sample performance')          
          }
          plot(LRvec, type = 'l', main = 'learning rate history')
          if(length(msevec)>1){
            plot(msevec[-1], type = 'l', main = 'MSE history')
            plot(msevec[pmax(2, length(msevec)-100):length(msevec)], type = 'l', ylab = 'mse', main = 'Last 100')
            plot(lossvec[-1], type = 'l', main = 'Loss history')
            plot(lossvec[pmax(2, length(lossvec)-100):length(lossvec)], type = 'l', ylab = 'loss', main = 'Last 100')
            if (!is.null(stop_early)){
              plot(mse_test_vec, type = "l", col = "blue", main = "Test MSE")
              plot(mse_test_vec[pmax(1, length(mse_test_vec)-100):length(mse_test_vec)], type = "l", col = "blue", main = "last 100")
            }
          }
        }
      } # fi verbose 
      if(iter > maxit | stopcounter > maxstopcounter){
        break
      }
    } #finishes epoch
  } #closes the while loop
  # revert to parlist_best
  if(return_best == TRUE){
    parlist <- parlist_best
    hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var,
                            nlayers = nlayers, convolutional = convolutional, activ = activation)
  }    
  # final values...
  yhat <- getYhat(parlist, hlay = hlayers, param, y, ydm, fe_var, nlayers) 
  mse <- mseold <- mean((y-yhat)^2)
  B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
  lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
  loss <- mse + lam * sum(c(parlist$beta_param*parapen,
                            B*as.numeric(penalize_toplayer),
                            lowerpar)^2)
  conv <- (iter < maxit)#Did we get convergence?
  if (is.null(fe_var)){
    fe_output <- NULL
  } else {
    # compute FE output
    Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
      hlayers[[i]][[length(hlayers[[i]])]]
    }
    Z <- cbind(param, as.matrix(Z))
    Zdm <- demeanlist(as.matrix(Z), list(fe_var))
    B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
    fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(parlist$beta_param, B)))
    fe_output <- data.frame(fe_var, fe)
  }
  output <- list(yhat = yhat, parlist = parlist, hidden_layers = hlayers
    , fe = fe_output, converged = conv, mse = mse, loss = loss, lam = lam, time_var = time_var
    , X = X, y = y, param = param, fe_var = fe_var, hidden_units = hidden_units, maxit = maxit
    , msevec = msevec, RMSprop = RMSprop, convtol = convtol
    , grads = grads, activation = activation, parapen = parapen
    , batchsize = batchsize, initialization = initialization, convolutional = convolutional
    , dropout_hidden = dropout_hidden, dropout_input = dropout_input, mse_test = best_mse
    , penalize_toplayer = penalize_toplayer)
  return(output) # list 
}






