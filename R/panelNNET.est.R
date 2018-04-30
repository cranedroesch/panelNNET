panelNNET.est <- function(y, X, hidden_units, fe_var, maxit, lam, time_var, param,
                          parapen, penalize_toplayer, parlist, verbose,
                          report_interval, gravity, convtol, RMSprop, start.LR,
                          activation, batchsize, maxstopcounter, OLStrick, OLStrick_interval,
                          initialization, dropout_hidden, dropout_input, convolutional,
                          LR_slowing_rate, return_best, stop_early, ...){

# rm(list=ls())
# gc()
# gc()
# "%ni%" <- Negate("%in%")
# mse <- function(x, y){mean((x-y)^2)}
# 
# library(devtools)
# install_github("cranedroesch/panelNNET", ref = "multinet", force = F)
# library(panelNNET)
# library(doParallel)
# library(doBy)
# library(glmnet)
# library(dplyr)
# library(randomForest)
# library(splines)
# 
# AWS <- grepl('ubuntu', getwd())
# desktop <- grepl(':', getwd())
# laptop <- grepl('/home/andrew', getwd())
# if(AWS){
#   setwd("/home/ubuntu/projdir")
#   outdir <- "/home/ubuntu/projdir/outdir"
#   registerDoParallel(detectCores())
# }
# if(desktop){
# }
# if(laptop){
#   setwd("/home/andrew/Dropbox/USDA/ARC/data")
#   outdir <- "/home/andrew/Dropbox/USDA/ARC/output"
#   registerDoParallel(detectCores())
# }
# dat <- readRDS("panel_corn.Rds")
# dat <- subset(dat, state %in% c("17", "19"))
# X1 <- dat[,paste0("maxat_jday_", 200:215)]
# X2 <- dat[,paste0("precip_jday_", 200:215)]
# dat$y <- dat$year - min(dat$year) + 1
# dat$y2 <- dat$y^2
# Xp <- dat[,c("y", "y2")]
# Xp <- Xp[sapply(Xp, sd) > 0]
# 
# is <- dat$year%%2==1
# oos <- is == F
# y = dat$yield[is]
# X = list(X1[is,], X2[is,])
# hidden_units = list(c(10, 2), c(8, 4))
# fe_var = dat$fips[is]
# maxit = 10000
# lam = 0
# time_var = dat$year[is]
# param = Xp[is,]
# verbose = T
# report_interval = 1
# gravity = 1.01
# convtol = 1e-3
# activation = 'lrelu'
# start.LR = .00001
# parlist = NULL
# OLStrick = T
# OLStrick_interval = 25
# batchsize = 256
# maxstopcounter = 2500
# LR_slowing_rate = 2
# parapen = c(0,0)
# penalize_toplayer = FALSE
# return_best = TRUE
# 
# RMSprop = T
# batchsize = 48
# initialization = "HZRS"
# dropout_hidden <- .5
# dropout_input <- .8
# convolutional <- NULL
# 
# # stop_early = list(check_every = 20,
# #                   max_ES_stopcounter = 5,
# #                   y_test = dat$yield[oos],
# #                   X_test = list(X1[oos,], X2[oos,]),
# #                   P_test = as.matrix(Xp[oos,]),
# #                   fe_test = dat$fips[oos])
# stop_early <- NULL


  ##########
  #Define internal functions

  recursive_add <- function(x, y) tryCatch(x + y, error = function(e) Map(recursive_add, x, y))
  recursive_mult <- function(x, y) tryCatch(x * y, error = function(e) Map(recursive_mult, x, y))
  recursive_RMSprop <- function(x, y) tryCatch(LR/sqrt(x+1e-10) * y, error = function(e) Map(recursive_RMSprop, x, y))
  
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
    if (length(nlayers)>1){
      if (!is.null(fe_var)){
        Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
          hlay[[i]][[length(hlay[[i]])]]
        }
        Z <- cbind(param, as.matrix(Z))
        Zdm <- demeanlist(Z, list(fe_var))
        B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
        fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(pl$beta_param, B)))
        yhat <- MatMult(Z, c(pl$beta_param, B)) + fe    
      } else {
        stop("not implement for no FE's")
      }            
    } else {
      if (!is.null(fe_var)){
        Zdm <- demeanlist(as.matrix(hlay[[length(hlay)]]), list(fe_var))
        fe <- (y-ydm) - MatMult(as.matrix(hlay[[length(hlay)]])-Zdm, as.matrix(c(pl$beta_param, pl$beta)))
        yhat <- MatMult(hlay[[length(hlay)]], c(pl$beta_param, pl$beta)) + fe    
      } else {
        yhat <- MatMult(hlay[[length(hlay)]], c(pl$beta_param, pl$beta))
      }      
    }
    return(as.numeric(yhat))
  }

  calc_grads<- function(plist, hlay = NULL, yhat = NULL, curBat = NULL, droplist = NULL, dropinp = NULL){
# plist <- parlist
# hlay <- hlayers
# curBat <- NULL
# droplist <- dropinp <- NULL
# hlay = hlay
# yhat = yhat[curBat]
# curBat = curBat
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
                                             droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])], 
                                             drop = FALSE]
      } else { #for one-layer networks
        #drop from parameter list emanating from input
        plist[[1]] <- plist[[1]][c(TRUE,dropinp),
                                 droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])], 
                                 drop = FALSE]
      }
      # manage parametric/nonparametric distinction in the top layer
      plist$beta <- plist$beta[droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])]]
    } else {Xd <- X}#for use below...  X should be safe given scope, but extra assignment is cheap here
    if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
    if (is.null(yhat)){yhat <- getYhat(plist, hlay = hlay)}
    NL <- nlayers + as.numeric(!is.null(convolutional))
    if (length(NL)>1){
      grads <- foreach(p = 1:length(NL)) %do% {
        grads <- grad_stubs <- vector('list', NL[p] + 1)
        grad_stubs[[length(grad_stubs)]] <- getDelta(CB(as.matrix(y)), yhat)
        for (i in NL[p]:1){
          # print(i)
          if (i == NL[p]){outer_param = as.matrix(c(plist[[p]]$beta))} else {outer_param = plist[[p]][[i+1]]}
          if (i == 1){lay = CB(Xd[[p]])} else {lay= CB(hlay[[p]][[i-1]])}
          #add the bias
          lay <- cbind(1, lay) #add bias to the hidden layer
          if (i != NL[p]){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
          grad_stubs[[i]] <- activ_prime(MatMult(lay, plist[[p]][[i]])) * MatMult(grad_stubs[[i+1]], Matrix::t(outer_param))
        }
        grad_stubs <- lapply(grad_stubs, as.matrix)
        for (i in 1:length(grad_stubs)){
          if (i == 1){lay = as.matrix(CB(Xd[[p]]))} else {lay= CB(hlay[[p]][[i-1]])}
          if (i != length(grad_stubs)){# don't add bias term to top layer when there are fixed effects present
            lay <- cbind(1, lay) #add bias to the hidden layer
          }
          grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
        }
        return(grads)
      }
      # add on parametric gradients
      grads[[length(grads)+1]] <- MatMult(t(CB(hlay$param)), getDelta(CB(as.matrix(y)), yhat)) 
    } else {
      grads <- grad_stubs <- vector('list', NL + 1)
      grad_stubs[[length(grad_stubs)]] <- getDelta(CB(as.matrix(y)), yhat)
      for (i in NL:1){
        # print(i)
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
        if (i != length(grad_stubs)){# don't add bias term to top layer when there are fixed effects present
          lay <- cbind(1, lay) #add bias to the hidden layer
        }
        grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
      } 
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
                               droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])]] <- grads[[nlayers]]
      } else { #for one-layer networks
        emptygrads[[1]][c(TRUE,dropinp),
                        droplist[[1]][(length(parlist$beta_param)+1):length(droplist[[1]])]] <- grads[[1]]
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
        rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- gg[,j]; x[1] <- -999; x[x!=0][-1]})
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
  if (class(X) == "list"){X <- lapply(X, scale)} else {X <- scale(X)}
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
    if (length(nlayers) >1){
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
    } else {
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
  if(is.null(fe_var) & !is.null(param)){
    parapen <- c(0, parapen)
  }
  if(is.null(fe_var) & is.null(param)){
    parapen <- 0
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
  # if using early stopping, initialize object for passing to predict function
  if (!is.null(stop_early)){
    if (is.null(fe_var)){
      fe_output = NULL; 
    } else {
      # compute FE output
      if(length(nlayers)>1){
        Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
          hlayers[[i]][[length(hlayers[[i]])]]
        }
        Zdm <- demeanlist(as.matrix(Z), list(fe_var))
        B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
        fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(parlist$beta_param, B)))
      } else {
        Zdm <- demeanlist(as.matrix(hlayers[[length(hlayers)]]), list(fe_var))
        Zdm <- Matrix(Zdm)
        fe <- (y-ydm) - MatMult(as.matrix(hlayers[[length(hlayers)]]-Zdm), as.matrix(c(
          parlist$beta_param, parlist$beta
        )))
      }
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
                                 newX = stop_early$X_test, 
                                 fe.newX = stop_early$fe_test, 
                                 new.param = stop_early$P_test)
    best_mse <- mse_test_vec <- mse_test <- msetest_old <- mean((stop_early$y_test-pr_test)^2)
  } else {mse_test <- best_mse <- NULL}
  # in-sample MSE and loss
  mse <- mseold <- mean((y-yhat)^2)
  if (length(nlayers)>1){
    B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
    lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
    loss <- mse + lam * sum(c(parlist$beta_param*parapen,
                              B*as.numeric(penalize_toplayer),
                              lowerpar)^2)
  } else {
    pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
    if (!is.null(convolutional)){
      pl_for_lossfun[[1]] <- unlist(c(convolutional$convParms, convolutional$convBias))
    }
    loss <- mse + lam*sum(c(parlist$beta_param*parapen 
                            , parlist$beta
                            , unlist(sapply(pl_for_lossfun, as.numeric)))^2
    )    
  }
  LRvec <- LR <- start.LR# starting step size
  #Calculate gradients
  grads <- calc_grads(parlist, hlayers, yhat, droplist = NULL, dropinp = NULL)
  #Initialize updates
  if (length(nlayers)>1){
    updates <- foreach(i = 1:length(nlayers))%do% {
      lapply(parlist[[i]], function(x){x*0})
    }
    updates$beta_param <- parlist$beta_param*0
  } else {
    updates <- lapply(parlist, function(x){x*0})
  }
  #initialize G2 term for RMSprop
  if (RMSprop == TRUE){
    #Prior gradients are zero at first iteration...
    G2 <- updates
    # #squashing all of the numeric list elements into a matrix/vector
    # betas <- matrix(unlist(G2[grepl('beta', names(G2))]))
    # G2 <- G2[!grepl('beta', names(G2))]
    # G2[[length(G2)+1]] <- betas
  } else {G2 <- NULL}
  # initialize terms used in the while loop
  D <- 1e6
  ES_stopcounter <- stopcounter <- iter <- 0
  msevec <- lossvec <- c()
  lossvec <- append(lossvec, loss)
  msevec <- append(msevec, mse)
  parlist_best <- parlist
  # 
  # # start with an OLStrick, before calculating the gradients
  # if (OLStrick == TRUE & (iter %% OLStrick_interval == 0 | iter == 0)){ # do OLStrick on first iteration
  #   # Update hidden layers
  #   hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var,
  #                           nlayers = nlayers, convolutional = convolutional, activ = activation)
  #   
  #   # OLS trick!
  #   parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
  #                                , fe_var = fe_var, lam = lam, parapen = parapen
  #                                , penalize_toplayer, nlayers = nlayers)
  # }
  ###############
  #start iterating
  while(iter < maxit & stopcounter < maxstopcounter){
    #Start epoch
    #Assign batches
    batchid <- sample(1:length(y) %/% batchsize +1)
    if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
      batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)), replace = TRUE)
    }
    for (bat in 1:max(batchid)) { # run minibatch
      iter <- iter + 1
      curBat <- which(batchid == bat)
      hlbatch <- calc_hlayers(parlist, X = lapply(X, function(X){X[curBat,]}), 
                              param = param[curBat,], fe_var = fe_var[curBat], 
                              nlayers = nlayers, convolutional = convolutional, activ = activation)
      # update hidden layers
      if (length(nlayers)>1){
        for (p in 1:length(nlayers)){
          for (h in 1:nlayers[p]){
            hlayers[[p]][[h]][curBat,] <- hlbatch[[p]][[h]]
          }                  
        }
      } else {
        for (h in 1:nlayers){
          hlayers[[h]][curBat,] <- hlbatch[[h]]
        }        
      }
      yhat <- getYhat(parlist, hlay = hlayers) # update yhat for purpose of computing gradients
      hlay <- hlayers# hlay may have experienced dropout, as distinct from hlayers
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
        droplist[[nlayers]][1:length(parlist$beta_param)] <- TRUE
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
      if (RMSprop ==  TRUE){
        oldG2 <- rapply(grads, function(x){.9*x^2}, how = "list")
      } 
      # Get updated gradients
      grads <- calc_grads(plist = parlist, hlay = hlay
        , yhat = yhat[curBat], curBat = curBat, droplist = droplist, dropinp = dropinp)
      # Calculate updates to parameters based on gradients and learning rates
      if (RMSprop == TRUE){
        newG2 <- rapply(grads, function(x){.1*x^2}, how = "list")
        G2 <- recursive_add(newG2, oldG2)
        # G2 <- mapply('+', newG2, oldG2)
        updates <- recursive_RMSprop(G2, grads)
        # uB <- LR/sqrt(G2[[length(G2)]]+1e-10) * grads[[length(grads)]]
        # updates$beta_param <- uB[1:length(parlist$beta_param)]
        # updates$beta <- uB[ifelse(is.null(param), 0, ncol(param))+(1:length(parlist$beta))]
        # # updates to lower layers
        # NL <- nlayers + as.numeric(!is.null(convolutional))
        # for(i in NL:1){
        #   updates[[i]] <- LR/sqrt(G2[[i]]+1e-10) * grads[[i]]
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
        parlist <- as.relistable(parlist)
        # don't update the pass-through weights for the non-time-varying variables when using conv 
        if (!is.null(convolutional)){
          updates[[1]][,colnames(updates[[1]]) %ni% convolutional$topology] <- 0
        }
      }
      # Update parameters from update list
      parlist <- relist(unlist(parlist) - unlist(updates))
      # parlist <- mapply('-', parlist, updates)
      if (OLStrick == TRUE & (iter %% OLStrick_interval == 0 | iter == 0)){ # do OLStrick on first iteration
        # Update hidden layers
        hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var,
                                nlayers = nlayers, convolutional = convolutional, activ = activation)
        # OLS trick!
        parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
                                     , fe_var = fe_var, lam = lam, parapen = parapen
                                     , penalize_toplayer, nlayers = nlayers)
      }
      #update yhat for purpose of computing loss function
      yhat <- getYhat(parlist, hlay = hlayers)
      mse <- mseold <- mean((y-yhat)^2)
      if (length(nlayers)>1){
        B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
        lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
        loss <- mse + lam * sum(c(parlist$beta_param*parapen,
                                  B*as.numeric(penalize_toplayer),
                                  lowerpar)^2)
      } else {
        pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
        if (!is.null(convolutional)){
          pl_for_lossfun[[1]] <- unlist(c(convolutional$convParms, convolutional$convBias))
        }
        loss <- mse + lam*sum(c(parlist$beta_param*parapen 
                                , parlist$beta
                                , unlist(sapply(pl_for_lossfun, as.numeric)))^2
        )    
      }
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
          if (is.null(fe_var)){
            fe_output <- NULL
          } else {
            # compute FE output
            if(length(nlayers)>2){
              Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
                hlay[[i]][[length(hlay[[i]])]]
              }
              Zdm <- demeanlist(as.matrix(Z), list(fe_var))
              B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
              fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(pl$beta_param, B)))
            } else {
              Zdm <- demeanlist(as.matrix(hlayers[[length(hlayers)]]), list(fe_var))
              Zdm <- Matrix(Zdm)
              fe <- (y-ydm) - MatMult(as.matrix(hlayers[[length(hlayers)]]-Zdm), as.matrix(c(
                parlist$beta_param, parlist$beta
              )))
            }
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
                                       newX = stop_early$X_test, 
                                       fe.newX = stop_early$fe_test, 
                                       new.param = stop_early$P_test)
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
    hlayers <- calc_hlayers(parlist, X = X, param = param,
                            fe_var = fe_var, nlayers = nlayers,
                            convolutional = convolutional, activ = activation)
  }    
  # final values...
  yhat <- getYhat(parlist, hlay = hlayers)
  mse <- mseold <- mean((y-yhat)^2)
  if (length(nlayers)>1){
    B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
    lowerpar <- foreach(i = 1:length(nlayers), .combine = c) %do% {unlist(parlist[[i]][1:nlayers[i]])}
    loss <- mse + lam * sum(c(parlist$beta_param*parapen,
                              B*as.numeric(penalize_toplayer),
                              lowerpar)^2)
  } else {
    pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
    if (!is.null(convolutional)){
      pl_for_lossfun[[1]] <- unlist(c(convolutional$convParms, convolutional$convBias))
    }
    loss <- mse + lam*sum(c(parlist$beta_param*parapen 
                            , parlist$beta
                            , unlist(sapply(pl_for_lossfun, as.numeric)))^2
    )    
  }
  conv <- (iter < maxit)#Did we get convergence?
  if (is.null(fe_var)){
    fe_output <- NULL
  } else {
    # compute FE output
    if(length(nlayers)>2){
      Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
        hlay[[i]][[length(hlay[[i]])]]
      }
      Zdm <- demeanlist(as.matrix(Z), list(fe_var))
      B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
      fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(pl$beta_param, B)))
    } else {
      Zdm <- demeanlist(as.matrix(hlayers[[length(hlayers)]]), list(fe_var))
      Zdm <- Matrix(Zdm)
      fe <- (y-ydm) - MatMult(as.matrix(hlayers[[length(hlayers)]]-Zdm), as.matrix(c(
        parlist$beta_param, parlist$beta
      )))
    }
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






