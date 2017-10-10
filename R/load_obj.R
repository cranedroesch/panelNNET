load_obj <-
function(f){
  env <- new.env()
  nm <- load(f, env)[1]
  env[[nm]]
}
