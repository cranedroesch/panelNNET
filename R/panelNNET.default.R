panelNNET.default <-
function(y, X, hidden_units, fe_var
  , maxit, lam = 0, time_var = NULL, param = NULL
  , parapen = rep(0, ncol(param)), penalize_toplayer = TRUE
  , parlist = NULL, verbose = FALSE
  , report_interval = 100
  , gravity = 1.01, convtol = 1e-8, RMSprop = TRUE, start_LR = .01
  , activation = 'relu'
  , batchsize = nrow(X)
  , maxstopcounter = 10, OLStrick = FALSE, OLStrick_interval = 1
  , initialization = 'HZRS'
  , dropout_hidden = 1, dropout_input = 1
  , convolutional = NULL, LR_slowing_rate = 2, return_best = TRUE, stop_early = NULL, ...)
{
  out <- panelNNET.est(y, X, hidden_units, fe_var, maxit, lam
    , time_var, param, parapen, penalize_toplayer, parlist, verbose
    , report_interval, gravity, convtol, RMSprop
    , start_LR, activation 
    , batchsize, maxstopcounter
    , OLStrick, OLStrick_interval, initialization, dropout_hidden, dropout_input
    , convolutional, LR_slowing_rate, return_best, stop_early
  )
  out$call = match.call()
  class(out) <- 'panelNNET'
  out
}
