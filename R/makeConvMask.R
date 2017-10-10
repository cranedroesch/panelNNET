
#function to make a mask for convolutional nets
# the purpose of the mask is to represent local connectivity
#topology argument should be an integer vector indicating positions in a 1-dimensional topology, with NA's for variables that aren't time-varying


makeMask <- function(X, topology, span, step, Nconv){
# topology <- convolutional$topology
# span <- convolutional$span <- 2
# step <- convolutional$step
# Nconv <- convolutional$Nconv
  stops <- seq(span+1, (max(topology, na.rm = T)-1), by = step)
  # make a matrix of zeros, of dimension equal to the number of inputs by the number of outputs (which is a function of the span)
  TVmask <- foreach(i = 1:length(topology), .combine = rbind) %do% {
    interval <- topology[i] + span * c(-1, 1)
    as.numeric(stops>= interval[1] & stops<=interval[2])
  }
  # First compute which rows of the time-varying mask are NAs
  NArows <- apply(TVmask, 1, function(x){any(is.na(x))})
  # Variables that don't have a topology should be NA -- they will get set to zero
  TVmask[is.na(TVmask)] <- 0
  colnames(TVmask) <- stops
  # If the last span overlaps the end of the topology, do a hack where the last span extends "upwards" into previous time periods
  # This will ensure that all columns have identical entries
  # the consequence of this is that the ending time periods are somewhat redundantly represented
  # first identify the position of the last entry in the last column
  endpos <- max(which(TVmask[,as.character(max(stops))] == 1))
  # then take the next column left, starting from last entry that is a 1
  endpos_neighbor <- max(which(TVmask[,as.character(stops[length(stops)-1])] == 1))
  neighbor <- TVmask[1:endpos_neighbor, as.character(stops[length(stops)-1])]
  # pad the front and back with zeros
  neighbor_repadded <- c(rep(0, endpos - endpos_neighbor), neighbor, rep(0, nrow(TVmask) - endpos))
  # and re-insert it
  TVmask[,as.character(max(stops))] <- neighbor_repadded
  # make a diagonal matrix for the non-TV terms
  NTVmask <- diag(rep(1, length(topology[is.na(topology)])))
  colnames(NTVmask) <- colnames(X)[is.na(topology)]
  # replicate the TVmask Nconv times
  TVmask <- t(do.call(rbind, replicate(Nconv, t(TVmask), simplify=FALSE)))
  # combine them.  first add on a zero matrix to the left of the TV matrix
  mask <- cbind(TVmask, 
                matrix(rep(0, ncol(NTVmask)*nrow(TVmask)), ncol = ncol(NTVmask))
                )
  # next add the NTVmask entries into the NArows
  mask[NArows,(ncol(TVmask)+1):ncol(mask)] <- NTVmask
  colnames(mask)[(ncol(TVmask)+1):ncol(mask)] <- colnames(NTVmask)
  rownames(mask) <- colnames(X)
  mask <- rbind(c(rep(1, ncol(TVmask)), rep(0, ncol(NTVmask))), mask)
  return(mask)
}

