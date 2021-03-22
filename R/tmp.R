rm(list = ls())
Rcpp::sourceCpp("src/hdsqr2.cpp")

library(MASS)
library(matrixStats)
library(glmnet)
library(caret)
library(rqPen)
library(conquer)
library(quantreg)

exam = function(beta, beta.hat) {
  TPR = sum(beta != 0 & beta.hat != 0) / sum(beta != 0)
  FPR = sum(beta == 0 & beta.hat != 0) / sum(beta == 0)
  err.itcp = abs(beta[1] - beta.hat[1])
  err.coef = norm(beta[-1] - beta.hat[-1], "2")
  return (c(TPR, FPR, err.itcp, err.coef))
}

geneCov = function(p) {
  rst = matrix(0, p, p)
  for (i in 1:(p - 1)) {
    for (j in i:p) {
      rst[i, j] = rst[j, i] = 0.7^(abs(i - j))
    }
  } 
  return (rst)
}

lambdaX = function(X,tau,B=2000) {
  Lam = numeric(B)
  for (b in 1:B) {
    Lam[b] = 2*max(abs(t(X) %*% (tau-(runif(n,0,1)<=tau))/ dim(X)[1]))
  }
  return(Lam)
}

qrloss = function(x,tau) {
  return( 0.5*abs(x)+(tau-0.5)*x )
}

n = 250
p = 800
s = 10
tau = 0.1
beta = c(rep(1.5, s+1), rep(0, p-s))
h = max(0.05, min((log(p)/n)^0.25, tau*(1-tau)/2))

X = matrix(rnorm(n*p),n,p)
Z = cbind(rep(1, n), X)
df = 2
err = rt(n, df) - qt(tau, df)
Y = Z %*% beta + err

lambda = max(lambdaX(Z,tau))
lambda = seq(0.1*lambda, lambda, length.out = 40)

# Solution path
Beta.qrlasso = matrix(0,p+1,40)
Beta.sqrlasso = matrix(0,p+1,40)
Beta.sqrscad = matrix(0,p+1,40)
Beta.sqrmcp = matrix(0,p+1,40)
model.size = matrix(0,4,40)
insample.pred = matrix(0,4,40)
score.max = matrix(0,4,40)

for (k in 1:40){
  Beta.qrlasso[,k] = rq(Y~X,tau,method="scad",lambda=lambda[k])$coef
  Beta.sqrlasso[,k] = SqrLasso(X, Y, lambda[k], tau, h, phi0 = 0.01, gamma = 1.5)
  Beta.sqrscad[,k] = SqrScad(X, Y, lambda[k], tau, h, phi0 = 0.01, gamma = 1.5)
  Beta.sqrmcp[,k] = SqrMcp(X, Y, lambda[k], tau, h, phi0 = 0.01, gamma = 1.5)
}


for (k in 1:40){
  model.size[1,k] = sum(abs(Beta.sqrlasso[-1,k])>0)
  model.size[2,k] = sum(abs(Beta.sqrscad[-1,k])>0)
  model.size[3,k] = sum(abs(Beta.sqrmcp[-1,k])>0)
  model.size[4,k] = sum(abs(Beta.qrlasso[-1,k])>0)
  
  lasso.res = Y-Z%*%Beta.sqrlasso[,k]
  scad.res = Y-Z%*%Beta.sqrscad[,k]
  mcp.res = Y-Z%*%Beta.sqrmcp[,k]
  qrlasso.res = Y-Z%*%Beta.qrlasso[,k]
  
  insample.pred[1,k] = mean(qrloss(lasso.res, tau))
  insample.pred[2,k] = mean(qrloss(scad.res, tau))
  insample.pred[3,k] = mean(qrloss(mcp.res, tau))
  insample.pred[4,k] = mean(qrloss(qrlasso.res, tau))
  
  score.max[1,k] = max(abs(t(Z)%*%(tau - (lasso.res<=0))/n))
  score.max[2,k] = max(abs(t(Z)%*%(tau - (scad.res<=0))/n))
  score.max[3,k] = max(abs(t(Z)%*%(tau - (mcp.res<=0))/n))
  score.max[4,k] = max(abs(t(Z)%*%(tau - (qrlasso.res<=0))/n))
}



start = Sys.time()
qr.lasso = rq(Y~X,tau,method="lasso",lambda=0.1)$coef
end = Sys.time()
print(c(end-start, sum((qr.lasso-beta)^2)))

start = Sys.time()
sqr.scad = SqrScad(X, Y, 0.1, tau, h, phi0 = 0.01, gamma = 1.5)
end = Sys.time()
print(c(end-start, sum((sqr.scad-beta)^2)))

########################## Comments ############################################

## Default number of outer loop iterations for SCAD/MCP: 3 or 4

## Use the element-wise max-norm of score function at the solution of each 
#  iteration to guard against divergence: stop the iteration once it goes up

## Lasso == SCAD/MCP with one outer loop iteration

## Change function names
#  conquer.lasso
#  conquer.scad + add choice of 2nd parameter (default = 3.7)
#  conquer.mcp + add choice of 2nd parameter (default = 3)

## Initial: asymmetric Lasso/Huber |tau - I(u<0)| * u^2/2 (Huber,C)

