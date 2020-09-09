# include <RcppArmadillo.h>
# include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
int sgn(const double x) {
  return (x > 0) - (x < 0);
}

/*// [[Rcpp::export]]
void updateUnif(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double h, 
                const double n1, const double h1) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
    } else if (cur < h) {
      der(i) = 0.5 - tau - 0.5 * h1 * cur;
    } else {
      der(i) = -tau;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updatePara(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double h, 
                const double n1, const double h1, const double h3) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
    } else if (cur < h) {
      der(i) = 0.5 - tau - 0.75 * h1 * cur + 0.25 * h3 * cur * cur * cur;
    } else {
      der(i) = -tau;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
void updateTrian(const arma::mat& Z, const arma::vec& res, arma::vec& der, arma::vec& grad, const int n, const double tau, const double h, 
                 const double n1, const double h1, const double h2) {
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    if (cur <= -h) {
      der(i) = 1 - tau;
    } else if (cur < 0) {
      der(i) = 0.5 - tau - h1 * cur - 0.5 * h2 * cur * cur;
    } else if (cur < h) {
      der(i) = 0.5 - tau - h1 * cur + 0.5 * h2 * cur * cur;
    } else {
      der(i) = -tau;
    }
  }
  grad = n1 * Z.t() * der;
}

// [[Rcpp::export]]
double sqLossUnif(const arma::vec& u, const int n, const double tau, const double h) {
  double rst = 0;
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur <= -h) {
      rst += (tau - 1) * cur;
    } else if (cur < h) {
      rst += cur * cur / (4 * h) + h / 4 + (tau - 0.5) * cur;
    } else {
      rst += tau * cur;
    }
  }
  return rst / n;
}

// [[Rcpp::export]]
double sqLossPara(const arma::vec& u, const int n, const double tau, const double h) {
  double rst = 0;
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur <= -h) {
      rst += (tau - 1) * cur;
    } else if (cur < h) {
      rst += 3 * h / 16 + 3 * cur * cur / (8 * h) - cur * cur * cur * cur / (16 * h * h * h) + (tau - 0.5) * cur;
    } else {
      rst += tau * cur;
    }
  }
  return rst / n;
}

// [[Rcpp::export]]
double sqLossTrian(const arma::vec& u, const int n, const double tau, const double h) {
  double rst = 0;
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur <= -h) {
      rst += (tau - 1) * cur;
    } else if (cur < 0) {
      rst += cur * cur * cur / (6 * h * h) + cur * cur / (2 * h) + h / 6 + (tau - 0.5) * cur;
    } else if (cur < h) {
      rst += -cur * cur * cur / (6 * h * h) + cur * cur / (2 * h) + h / 6 + (tau - 0.5) * cur;
    } else {
      rst += tau * cur;
    }
  }
  return rst / n;
}*/

// [[Rcpp::export]]
arma::mat center(arma::mat X, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) -= arma::mean(X.col(i));
  }
  return X;
}

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - arma::mean(X.col(i))) / arma::stddev(X.col(i));
  }
  return X;
}

// [[Rcpp::export]]
arma::vec softThresh(const arma::vec& x, const arma::vec& lambda, const int p) {
  return arma::sign(x) % arma::max(arma::abs(x) - lambda, arma::zeros(p + 1));
}

// [[Rcpp::export]]
arma::vec cmptLambdaLasso(const arma::vec& beta, const double lambda, const int p) {
  arma::vec rst = lambda * arma::ones(p + 1);
  rst(0) = 0;
  return rst;
}

// [[Rcpp::export]]
arma::vec cmptLambdaSCAD(const arma::vec& beta, const double lambda, const int p) {
  arma::vec rst(p + 1);
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    if (abBeta <= lambda) {
      rst(i) = lambda;
    } else if (abBeta <= 3.7 * lambda) {
      rst(i) = 0.37037 * (3.7 * lambda - abBeta);
    } else {
      rst(i) = 0;
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec cmptLambdaMCP(const arma::vec& beta, const double lambda, const int p) {
  arma::vec rst(p + 1);
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    rst(i) = (abBeta <= 3 * lambda) ? (lambda - 0.33333 * abBeta) : 0;
  }
  return rst;
}

// For l_2 loss, update gradient, return loss
// [[Rcpp::export]]
double lossL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double n1) {
  arma::vec res = Z * beta - Y;
  return 0.5 * arma::mean(arma::square(res));
}

// [[Rcpp::export]]
double updateL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double n1) {
  arma::vec res = Z * beta - Y;
  grad = n1 * Z.t() * res;
  return 0.5 * arma::mean(arma::square(res));
}

// For sqr loss, update gradient, return loss
// [[Rcpp::export]]
double lossGauss(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1, const double h2) {
  arma::vec res = Z * beta - Y;
  arma::vec temp = 0.39894 * h  * arma::exp(-0.5 * h2 * arma::square(res)) - tau * res + res % arma::normcdf(h1 * res);
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateGauss(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                   const double h1, const double h2) {
  arma::vec res = Z * beta - Y;
  arma::vec der = arma::normcdf(res * h1) - tau;
  grad = n1 * Z.t() * der;
  arma::vec temp = 0.39894 * h  * arma::exp(-0.5 * h2 * arma::square(res)) - tau * res + res % arma::normcdf(h1 * res);
  return arma::mean(temp);
}

// LAMM, update beta, return phi
// [[Rcpp::export]]
double lammL2(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double gamma, const int p, 
              const double n1) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateL2(Z, Y, beta, grad, n1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossL2(Z, Y, betaNew, n1);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammSq(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double phi, const double tau, 
              const double gamma, const int p, const double h, const double n1, const double h1, const double h2) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad(p + 1);
  double loss = updateGauss(Z, Y, beta, grad, tau, n1, h, h1, h2);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossGauss(Z, Y, betaNew, tau, h, h1, h2);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
arma::vec lasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const int p, const double n1, const double phi0 = 0.01, 
                const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = arma::zeros(p + 1);
  arma::vec betaNew = arma::zeros(p + 1);
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammL2(Z, Y, Lambda, betaNew, phi, gamma, p, n1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec scad(const arma::mat& Z, const arma::vec& Y, const double lambda, const int p, const double n1, const double phi0 = 0.01, 
               const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = arma::zeros(p + 1);
  arma::vec betaNew = arma::zeros(p + 1);
  // Contraction
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammL2(Z, Y, Lambda, betaNew, phi, gamma, p, n1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0(p + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, p);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammL2(Z, Y, Lambda, betaNew, phi, gamma, p, n1);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec mcp(const arma::mat& Z, const arma::vec& Y, const double lambda, const int p, const double n1, const double phi0 = 0.01, 
              const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = arma::zeros(p + 1);
  arma::vec betaNew = arma::zeros(p + 1);
  // Contraction
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammL2(Z, Y, Lambda, betaNew, phi, gamma, p, n1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0(p + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, p);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammL2(Z, Y, Lambda, betaNew, phi, gamma, p, n1);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec sqrLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                   const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, 
                   const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSq(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec sqrScad(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                  const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, 
                  const int iteMax = 500) {
  arma::vec beta = scad(Z, Y, lambda, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSq(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0(p + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, p);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammSq(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec sqrMcp(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                 const double h1, const double h2, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, 
                 const int iteMax = 500) {
  arma::vec beta = mcp(Z, Y, lambda, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammSq(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0(p + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, p);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammSq(Z, Y, Lambda, betaNew, phi, tau, gamma, p, h, n1, h1, h2);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec cvSqrLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                     const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = Z.n_rows, p = Z.n_cols - 1, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    for (int i = 0; i < nlambda; i++) {
      betaHat = sqrLasso(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(arma::square(testY - testZ * betaHat));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  return sqrLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
}

// [[Rcpp::export]]
arma::vec cvSqrScad(const arma::mat& Z, const arma::vec& Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                    const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = Z.n_rows, p = Z.n_cols - 1, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    for (int i = 0; i < nlambda; i++) {
      betaHat = sqrScad(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(arma::square(testY - testZ * betaHat));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  return sqrScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
}

// [[Rcpp::export]]
arma::vec cvSqrMcp(const arma::mat& Z, const arma::vec& Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                   const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = Z.n_rows, p = Z.n_cols - 1, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, h2 = 1.0 / (h * h);
  arma::vec betaHat(p + 1);
  arma::vec mse = arma::zeros(nlambda);
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    for (int i = 0; i < nlambda; i++) {
      betaHat = sqrMcp(trainZ, trainY, lambdaSeq(i), tau, p, n1Train, h, h1, h2, phi0, gamma, epsilon, iteMax);
      mse(i) += arma::accu(arma::square(testY - testZ * betaHat));
    }
  }
  arma::uword cvIdx = arma::index_min(mse);
  return sqrMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, h2, phi0, gamma, epsilon, iteMax);
}
