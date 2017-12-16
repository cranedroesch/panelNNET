panelNNET.est <-
function(y, X, hidden_units, fe_var, maxit, lam, time_var, param, parapen, parlist
         , verbose, report_interval, gravity, convtol, RMSprop
         , start.LR, activation
         , batchsize, maxstopcounter, OLStrick, initialization, dropout_hidden
         , dropout_input, convolutional, LR_slowing_rate, ...){

  
  # y = dat$logyield[dat$year %in% samp]
  # X = X[dat$year %in% samp,]
  # hidden_units = rep(100, 10)
  # fe_var = dat$fips[dat$year %in% samp]
  # maxit = 300
  # lam = lam
  # time_var = dat$year[dat$year %in% samp]
  # param = Xp[dat$year %in% samp,]
  # verbose = F
  # report_interval = 1
  # gravity = 1.1
  # convtol = 1e-4
  # activation = 'lrelu'
  # start_LR = .001
  # parlist = parlist 
  # OLStrick = TRUE
  # batchsize = batchsize
  # maxstopcounter = 10
  # dropout_hidden = D
  # dropout_input = D^(log(.8)/log(.5))
  # parapen = c(0,0,rep(1, ncol(Xp)-2))
  # initialization = 'HZRS'
  # RMSprop = T
  # start.LR <- .001
  # maxstopcounter <- 10
  # batchsize = round(nrow(X)/100)
  # convolutional <- NULL
  # parapen <- rep(1, ncol(Xp))
  # LR_slowing_rate <- 2
  # gravity = 1.1
  
  ##########
  #Define internal functions
  getYhat <- function(pl, hlay = NULL){ 
    #Update hidden layers
    if (is.null(hlay)){hlay <- calc_hlayers(pl,
                                            X = X,
                                            param = param,
                                            fe_var = fe_var,
                                            nlayers = nlayers,
                                            convolutional = convolutional,
                                            activ = activation)}
    #update yhat
    if (!is.null(fe_var)){
      Zdm <- demeanlist(as.matrix(hlay[[length(hlay)]]), list(fe_var))
      fe <- (y-ydm) - MatMult(as.matrix(hlay[[length(hlay)]])-Zdm, as.matrix(c(pl$beta_param, pl$beta)))
      yhat <- MatMult(hlay[[length(hlay)]], c(pl$beta_param, pl$beta)) + fe    
    } else {
      yhat <- MatMult(hlay[[length(hlay)]], c(pl$beta_param, pl$beta))
    }
    return(as.numeric(yhat))
  }

  calc_grads<- function(plist, hlay = NULL, yhat = NULL, curBat = NULL, droplist = NULL, dropinp = NULL){
    #subset the parameters and hidden layers based on the droplist
    if (!is.null(droplist)){
      Xd <- X[,dropinp, drop = FALSE]
      if (nlayers > 1){
        #drop from parameter list emanating from input
        plist[[1]] <- plist[[1]][c(TRUE,dropinp),droplist[[1]]]
        # drop from subsequent parameter matrices
        if (nlayers>2){
          for (i in 2:(nlayers-1)){
            plist[[i]] <- plist[[i]][c(TRUE, droplist[[i-1]]), droplist[[i]], drop = FALSE]
          }
        }
        plist[[nlayers]] <- plist[[nlayers]][c(TRUE, droplist[[nlayers-1]]), 
                                             droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])], 
                                             drop = FALSE]
      } else { #for one-layer networks
        #drop from parameter list emanating from input
        plist[[1]] <- plist[[1]][c(TRUE,dropinp),
                                 droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])], 
                                 drop = FALSE]
      }
      # manage parametric/nonparametric distinction in the top layer
      plist$beta <- plist$beta[droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])]]
      
    } else {Xd <- X}#for use below...  X should be safe given scope, but extra assignment is cheap here
    if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
    if (is.null(yhat)){yhat <- getYhat(plist, hlay = hlay)}
    NL <- nlayers + as.numeric(!is.null(convolutional))
    grads <- grad_stubs <- vector('list', NL + 1)
    grad_stubs[[length(grad_stubs)]] <- getDelta(CB(as.matrix(y)), yhat)
    for (i in NL:1){
      if (i == NL){outer_param = as.matrix(c(plist$beta))} else {outer_param = plist[[i+1]]}
      if (i == 1){lay = CB(Xd)} else {lay= CB(hlay[[i-1]])}
      #add the bias
      lay <- cbind(1, lay) #add bias to the hidden layer
      if (i != NL){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
      grad_stubs[[i]] <- activ_prime(MatMult(lay, plist[[i]])) * MatMult(grad_stubs[[i+1]], Matrix::t(outer_param))
    }
    # multiply the gradient stubs by their respective layers to get the actual gradients
    # first coerce them to regular matrix classes so that the C code for matrix multiplication can speed things up
    grad_stubs <- lapply(grad_stubs, as.matrix)
    hlay <- lapply(hlay, as.matrix)
    for (i in 1:length(grad_stubs)){
      if (i == 1){lay = as.matrix(CB(Xd))} else {lay= CB(hlay[[i-1]])}
      if (i != length(grad_stubs) | is.null(fe_var)){# don't add bias term to top layer when there are fixed effects present
        lay <- cbind(1, lay) #add bias to the hidden layer
      }
      grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
    }
    # if using dropout, reconstitute full gradient
    if (!is.null(droplist)){
      emptygrads <- lapply(parlist, function(x){x*0})
      # bottom weights
      if (nlayers > 1){
        emptygrads[[1]][c(TRUE,dropinp),droplist[[1]]] <- grads[[1]]
        if (nlayers>2){
          for (i in 2:(nlayers-1)){
            emptygrads[[i]][c(TRUE, droplist[[i-1]]), droplist[[i]]] <- grads[[i]]
          }
        }
        emptygrads[[nlayers]][c(TRUE, droplist[[nlayers-1]]), 
                               droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])]] <- grads[[nlayers]]
      } else { #for one-layer networks
        emptygrads[[1]][c(TRUE,dropinp),
                        droplist[[1]][(ncol(param)+1):length(droplist[[1]])]] <- grads[[1]]
      }
      #top-level
      emptygrads$beta <- emptygrads$beta_param <- NULL
      emptygrads[[nlayers + 1]] <- matrix(rep(0, length(parlist$beta)+length(parlist$beta_param))) #empty
      emptygrads[[nlayers + 1]][droplist[[nlayers]]] <- grads[[nlayers + 1]]
      # all done
      grads <- emptygrads
    }
    #process the gradients for the convolutional layers
    if (!is.null(convolutional)){
      if (!is.null(droplist)){
        warning("dropout not yet made to work with conv nets")
      }
      #mask out the areas not in use
      gg <- grads[[1]] * convMask
      #gradients for conv layer.  pooling via rowMeans
      grads_convParms <- foreach(i = 1:convolutional$Nconv) %do% {
        idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
        rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- gg[,j]; x[x!=0][-1]})
      }
      grads_convBias <- foreach(i = 1:convolutional$Nconv, .combine = c) %do% {
        idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
        mean(gg[1,idx])
      }
      # make the layer
      convGrad <- makeConvLayer(grads_convParms, grads_convBias)
      #set the gradients on the time-invariant terms to zero
      convGrad[,(N_TV_layers * convolutional$Nconv+1):ncol(convGrad)] <- 0
      grads[[1]] <- convGrad
    }
    return(grads)
  }

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
  
  ###########################
  # sanity checks.  here place checks to ensure that arguments supplied will yield sensible output
  ###########################
  if (gravity <= 1){stop("Gravity must be >1")}
  if (start.LR <= 0){stop("Learning rate must be positive")}
  # if (LR_slowing_rate <= 1){stop("LR_slowing_rate must larger than 1")}
  ###########################
  # start fitting
  ###########################
  # do scaling
  X <- scale(X)
  if (!is.null(param)){
    param <- scale(param)
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
  nlayers <- length(hidden_units)
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
    parlist <- vector('list', nlayers)
    for (i in 1:nlayers){
      if (i == 1){
        if (is.null(convolutional)){
          D <- ncol(X)
        } else {
          D <- ncol(convolutional$convParMat)
        }
      } else {
        D <- hidden_units[i-1]
      }
      if (initialization %ni% c('XG', 'HZRS')){#random initialization schemes
        ubounds <- .7 #follows ESL recommendaton
      } else {
        if (initialization == 'XG'){
          ubounds <- sqrt(6)/sqrt(D+hidden_units[i]+2)#2 is for the bias.  Not sure why 2.  Would need to go back and read the paper.  
        }
        if (initialization == 'HZRS'){
          ubounds <- 2*sqrt(6)/sqrt(D+hidden_units[i]+2)#2 is for the bias.  Not sure why 2.  Would need to go back and read the paper.
        }
      }
      parlist[[i]] <- matrix(runif((hidden_units[i])*(D+1), -ubounds, ubounds), ncol = hidden_units[i])
    }
    # vector of parameters at the top layer
    parlist$beta <- runif(hidden_units[i], -ubounds, ubounds)
    # add convolutional layer on the bottom
    if (!is.null(convolutional)){
      parlist <- c(convolutional$convParMat, parlist)
    }
    # parameters on parametric terms
    if (is.null(param)){
      parlist$beta_param <-  NULL
    } else {
      parlist$beta_param <- runif(ncol(param), -ubounds, ubounds)
    }
    #add the bias term/intercept onto the front, if there are no FE's
    parlist$beta_param <- c(runif(is.null(fe_var), -ubounds, ubounds), parlist$beta_param)
    #if there are no FE's, append a 0 to the front of the parapen vec, to leave the intercept unpenalized
    if(is.null(fe_var)){
      parapen <- c(0, parapen)
    }
  }
  #compute hidden layers given parlist
  hlayers <- calc_hlayers(parlist, X = X, param = param, 
                          fe_var = fe_var, nlayers = nlayers, 
                          convolutional = convolutional, activation = activation)
  #calculate ydm and put it in global...
  if (!is.null(fe_var)){
    ydm <<- demeanlist(y, list(fe_var)) 
  }
  #####################################
  #start setup
  #get starting mse
  yhat <- as.numeric(getYhat(parlist, hlay = hlayers))
  mse <- mseold <- mean((y-yhat)^2)
  pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
  if (!is.null(convolutional)){
    pl_for_lossfun[[1]] <- unlist(c(convolutional$convParms, convolutional$convBias))
  }
  loss <- mse + lam*sum(c(parlist$beta_param*parapen 
    , parlist$beta
    , unlist(sapply(pl_for_lossfun, as.numeric)))^2
  )
  LRvec <- LR <- start.LR# starting step size
  #Calculate gradients
  grads <- calc_grads(parlist, hlayers, yhat, droplist = NULL, dropinp = NULL)
  #Initialize updates
  updates <- lapply(parlist, function(x){x*0})
  #initialize G2 term for RMSprop
  if (RMSprop == TRUE){
    #Prior gradients are zero at first iteration...
    G2 <- lapply(parlist, function(x){x*0})
    #squashing all of the numeric list elements into a matrix/vector
    betas <- matrix(unlist(G2[grepl('beta', names(G2))]))
    G2 <- G2[!grepl('beta', names(G2))]
    G2[[length(G2)+1]] <- betas
  } else {G2 <- NULL}
  # initialize terms used in the while loop
  D <- 1e6
  stopcounter <- iter <- 0
  msevec <- lossvec <- c()
  lossvec <- append(lossvec, loss)
  msevec <- append(msevec, mse)
  parlist_best <- parlist
  ###############
  #start iterating
  while(iter < maxit & stopcounter < maxstopcounter){
    #Start epoch
    #Assign batches
    batchid <- sample(1:nrow(X) %/% batchsize +1)
    if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
      batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)), replace = TRUE)
    }
    for (bat in 1:max(batchid)) { # run minibatch
      iter <- iter + 1
      curBat <- which(batchid == bat)
      hlay <- hlayers#h lay may have experienced dropout, as distinct from hlayers
      # if using dropout, generate a droplist
      if (dropout_hidden < 1){
        droplist <- lapply(hlayers, function(x){
          todrop <- as.logical(rbinom(ncol(x), 1, dropout_hidden))
          if (all(todrop==FALSE)){#ensure that at least one unit is present
            todrop[sample(1:length(todrop))] <- TRUE
          }
          return(todrop)
        })
        # remove the parametric terms from dropout contention
        droplist[[nlayers]][1:ncol(param)] <- TRUE
        # dropout from the input layer
        todrop <- rbinom(ncol(X), 1, dropout_input)
        if (all(todrop==FALSE)){# ensure that at least one unit is present
          todrop[sample(1:length(todrop))] <- TRUE
        }
        dropinp <- as.logical(todrop)
        for (i in 1:nlayers){
          hlay[[i]] <- hlay[[i]][,droplist[[i]], drop = FALSE]
        }
        Xd <- X[,dropinp]
      } else {Xd <- X; droplist = NULL}
      # before updating gradients, compute square of gradients for RMSprop
      if (RMSprop ==  TRUE){oldG2 <- lapply(grads, function(x){.9*x^2})} #old G2 term 
      # Get updated gradients
      grads <- calc_grads(plist = parlist, hlay = hlay
        , yhat = yhat[curBat], curBat = curBat, droplist = droplist, dropinp = dropinp)
      # Calculate updates to parameters based on gradients and learning rates
      if (RMSprop == TRUE){
        newG2 <- lapply(grads, function(x){.1*x^2}) #new gradient is squared and multiplied by .1
        G2 <- mapply('+', newG2, oldG2)
        # updates to beta
        uB <- LR/sqrt(G2[[length(G2)]]+1e-10) * grads[[length(grads)]]
        updates$beta_param <- uB[1:length(parlist$beta_param)]
        updates$beta <- uB[ncol(param)+(1:length(parlist$beta))]
        # updates to lower layers
        NL <- nlayers + as.numeric(!is.null(convolutional))
        for(i in NL:1){
          updates[[i]] <- LR/sqrt(G2[[i]]+1e-10) * grads[[i]]
        }
      } else { #if RMSprop == FALSE
        uB <- LR * grads[[length(grads)]]
        updates$beta_param <- uB[1:length(parlist$beta_param)]
        updates$beta <- uB[ncol(param)+(1:length(parlist$beta))]
        NL <- nlayers + as.numeric(!is.null(convolutional))
        for(i in NL:1){
          updates[[i]] <- LR * grads[[i]]
        }
      }
      # weight decay
      if (lam != 0) {
        wd <- lapply(parlist, function(x){x*lam*LR})
        updates <- mapply("+", updates, wd)
        # don't update the pass-through weights for the non-time-varying variables when using conv 
        if (!is.null(convolutional)){
          updates[[1]][,colnames(updates[[1]]) %ni% convolutional$topology] <- 0
        }
      }
      # Update parameters from update list
      parlist <- mapply('-', parlist, updates)
      # Update hidden layers
      hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var, 
                              nlayers = nlayers, convolutional = convolutional, activ = activation)
      # OLS trick!
      if (OLStrick == TRUE){
        parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
                                     , fe_var = fe_var, lam = lam, parapen = parapen)
      }

      #update yhat
      yhat <- getYhat(parlist, hlay = hlayers)
      mse <- mean((y-yhat)^2)
      pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
      if (!is.null(convolutional)){ # coerce the convolutional parameters to a couple of vectors to avoid double-counting in the loss
        convolutional$convParms <- foreach(i = 1:convolutional$Nconv) %do% {
          idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
          rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- pl_for_lossfun[[1]][,j]; x[x!=0][-1]})
        }
        convolutional$convBias <- foreach(i = 1:convolutional$Nconv, .combine = c) %do% {
          idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
          mean(pl_for_lossfun[[1]][1,idx])
        }
        pl_for_lossfun[[1]] <- c(unlist(convolutional$convParms, convolutional$convBias))
      }
      loss <- mse + lam*sum(c(parlist$beta_param*parapen
                              , parlist$beta
                              , unlist(sapply(pl_for_lossfun, as.numeric)))^2
      )
      oldloss <- lossvec[length(lossvec)]
      oldmse <- msevec[length(msevec)]
      lossvec <- append(lossvec, loss)
      msevec <- append(msevec, mse)
      # if achieving a new minimum, stash parlist in parlist_best
      if (loss == min(lossvec)){
        parlist_best <- parlist
      }
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
          par(mfrow = c(3,2))
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
          }
        }
      } # fi verbose 
      if(iter > maxit | stopcounter > maxstopcounter){
        break
      }
    } #finishes epoch
  } #closes the while loop
  # revert to parlist_best
  parlist <- parlist_best
  hlayers <- calc_hlayers(parlist, X = X, param = param,
                          fe_var = fe_var, nlayers = nlayers,
                          convolutional = convolutional, activ = activation)
  # #If trained with dropput, weight the layers by expectations
  # if(dropout_hidden<1){
  #   for (i in nlayers:1){
  #     if (i == 1){
  #       parlist[[i]] <- parlist[[i]] * dropout_input
  #     } else {
  #       parlist[[i]] <- parlist[[i]] * dropout_hidden
  #     }
  #   }
  #   parlist$beta <- parlist$beta * dropout_hidden
  #   if (OLStrick == TRUE){
  #     parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
  #       , fe_var = fe_var, lam = lam, parapen = parapen)
  #   }
  #   #redo the hidden layers based on the new parlist
  #   hlayers <- calc_hlayers(parlist, X = X, param = param,
  #                           fe_var = fe_var, nlayers = nlayers,
  #                           convolutional = convolutional, activ = activation)
  # }
  # final values...
  yhat <- getYhat(parlist, hlay = hlayers)
  mse <- mean((y-yhat)^2)
  pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
  if (!is.null(convolutional)){ # coerce the convolutional parameters to a couple of vectors to avoid double-counting in the loss
    convolutional$convParms <- foreach(i = 1:convolutional$Nconv) %do% {
      idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
      rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- pl_for_lossfun[[1]][,j]; x[x!=0][-1]})
    }
    convolutional$convBias <- foreach(i = 1:convolutional$Nconv, .combine = c) %do% {
      idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
      mean(pl_for_lossfun[[1]][1,idx])
    }
    pl_for_lossfun[[1]] <- c(unlist(convolutional$convParms, convolutional$convBias))
  }
  loss <- mse + lam*sum(c(parlist$beta_param*parapen
                          , parlist$beta
                          , unlist(sapply(pl_for_lossfun, as.numeric)))^2
  )
  conv <- (iter < maxit)#Did we get convergence?
  if(is.null(fe_var)){
    fe_output <- NULL
  } else {
    Zdm <- demeanlist(as.matrix(hlayers[[length(hlayers)]]), list(fe_var))
    Zdm <- Matrix(Zdm)
    fe <- (y-ydm) - as.matrix(hlayers[[length(hlayers)]]-Zdm) %*% as.matrix(c(
        parlist$beta_param, parlist$beta
    ))
  fe_output <- data.frame(fe_var, fe)
  }
  output <- list(yhat = yhat, parlist = parlist, hidden_layers = hlayers
    , fe = fe_output, converged = conv, mse = mse, loss = loss, lam = lam, time_var = time_var
    , X = X, y = y, param = param, fe_var = fe_var, hidden_units = hidden_units, maxit = maxit
    , msevec = msevec, RMSprop = RMSprop, convtol = convtol
    , grads = grads, activation = activation, parapen = parapen
    , batchsize = batchsize, initialization = initialization, convolutional = convolutional
    , dropout_hidden = dropout_hidden, dropout_input = dropout_input)
  return(output) # list 
}






