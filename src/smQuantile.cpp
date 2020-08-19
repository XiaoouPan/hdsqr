# include <RcppArmadillo.h>
# include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
int sgn(const double x) {
  return (x > 0) - (x < 0);
}

// [[Rcpp::export]]
double mad(const arma::vec& x) {
  return arma::median(arma::abs(x - arma::median(x))) / 0.6744898;
}

// [[Rcpp::export]]
double quant(const arma::vec& x, const int n, const double tau, const double tol = 0.0001) {
  double low = x.min(), high = x.max();
  while (high - low > tol) {
    double mid = (high + low) / 2;
    double cur = (double)arma::accu(x <= mid) / n;
    if (cur == tau) {
      return mid;
    } else if (cur < tau) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return (high + low) / 2;
}

// [[Rcpp::export]]
arma::vec huberDer(const arma::vec& x, const int n, const double tau) {
  arma::vec w(n);
  for (int i = 0; i < n; i++) {
    w(i) = std::abs(x(i)) <= tau ? -x(i) : -tau * sgn(x(i));
  }
  return w;
}

// [[Rcpp::export]]
double huberLoss(const arma::vec& x, const int n, const double tau) {
  double loss = 0;
  for (int i = 0; i < n; i++) {
    double cur = x(i);
    loss += std::abs(cur) <= tau ? (cur * cur / 2) : (tau * std::abs(cur) - tau * tau / 2);
  }
  return loss / n;
}

// [[Rcpp::export]]
arma::vec huberReg(const arma::mat& Z, const arma::vec& Y, const int n, const int p, 
                   const double tol = 0.00001, const double constTau = 1.345, const int iteMax = 500) {
  arma::vec betaOld = arma::zeros(p + 1);
  double tau = constTau * mad(Y);
  arma::vec gradOld = Z.t() * huberDer(Y, n, tau) / n;
  double lossOld = huberLoss(Y - Z * betaOld, n, tau);
  arma::vec betaNew = betaOld - gradOld;
  arma::vec res = Y - Z * betaNew;
  double lossNew = huberLoss(res, n, tau);
  arma::vec gradNew, gradDiff, betaDiff;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaNew - betaOld, "inf") > tol && ite <= iteMax) {
    tau = constTau * mad(res);
    gradNew = Z.t() * huberDer(res, n, tau) / n;
    gradDiff = gradNew - gradOld;
    betaDiff = betaNew - betaOld;
    double alpha = 1.0;
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), 1.0);
    }
    betaOld = betaNew;
    gradOld = gradNew;
    lossOld = lossNew;
    betaNew -= alpha * gradNew;
    res += alpha * Z * gradNew; 
    lossNew = huberLoss(res, n, tau);
    ite++;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec sqDerGauss(const arma::vec& u, const double tau) {
  return arma::normcdf(u) - tau;
}

// [[Rcpp::export]]
arma::vec sqDerUnif(const arma::vec& u, const int n, const double tau) {
  arma::vec rst(n);
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur > 1) {
      rst(i) = 1 - tau;
    } else if (cur > -1) {
      rst(i) = (1 + cur) / 2 - tau;
    } else {
      rst(i) = -tau;
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec sqDerPara(const arma::vec& u, const int n, const double tau) {
  arma::vec rst(n);
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur > 1) {
      rst(i) = 1 - tau;
    } else if (cur > -1) {
      rst(i) = (3 * cur - cur * cur * cur + 2) / 4 - tau;
    } else {
      rst(i) = -tau;
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec sqDerTrian(const arma::vec& u, const int n, const double tau) {
  arma::vec rst(n);
  for (int i = 0; i < n; i++) {
    double cur = u(i);
    if (cur > 1) {
      rst(i) = 1 - tau;
    } else if (cur > 0) {
      rst(i) = (2 * cur - cur * cur + 1) / 2 - tau;
    } else if (cur > -1) {
      rst(i) = (1 + cur * cur + 2 * cur) / 2 - tau;
    } else {
      rst(i) = -tau;
    }
  }
  return rst;
}

// [[Rcpp::export]]
double sqLossGauss(const arma::vec& u, const double tau, const double h) {
  arma::vec temp = h * 0.79788 * arma::exp(-arma::square(u) / (2 * h * h)) + u - 2 * u % arma::normcdf(-u / h);
  return arma::mean(temp / 2 + (tau - 0.5) * u);
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
}

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
Rcpp::List smqrGauss(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, 
                     const double constTau = 1.345, const double tol = 0.00001, const int iteMax = 500) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::sqrt(tau * (1 - tau)) * std::cbrt((double)(p + 1) / n);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  double c0 = 4 * tau * (1 - tau);
  arma::vec betaOld = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  betaOld(0) = quant(Y - Z.cols(1, p) * betaOld.rows(1, p), n, tau);
  arma::vec res = Y - Z * betaOld;
  double lossOld = sqLossGauss(res, tau, h);
  arma::vec gradOld = Z.t() * sqDerGauss(-res / h, tau) / n;
  arma::vec betaNew = betaOld - c0 * gradOld;
  res = Y - Z * betaNew;
  double lossNew = sqLossGauss(res, tau, h);
  arma::vec gradNew, gradDiff, betaDiff;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaNew - betaOld, "inf") > tol && ite <= iteMax) {
    gradNew = Z.t() * sqDerGauss(-res / h, tau) / n;
    gradDiff = gradNew - gradOld;
    betaDiff = betaNew - betaOld;
    double alpha = c0 / std::pow(ite, std::abs(tau - 0.5));
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), c0) / std::pow(ite, std::abs(tau - 0.5));
    }
    betaOld = betaNew;
    gradOld = gradNew;
    lossOld = lossNew;
    betaNew -= alpha * gradNew;
    res += alpha * Z * gradNew;
    lossNew = sqLossGauss(res, tau, h);
    ite++;
  }
  betaNew.rows(1, p) /= arma::stddev(X, 0, 0).t();
  betaNew(0) -= arma::as_scalar(arma::mean(X, 0) * betaNew.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaNew, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
Rcpp::List smqrUnif(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, 
                    const double constTau = 1.345, const double tol = 0.00001, const int iteMax = 500) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::sqrt(tau * (1 - tau)) * std::cbrt((double)(p + 1) / n);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  double c0 = 4 * tau * (1 - tau);
  arma::vec betaOld = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  betaOld(0) = quant(Y - Z.cols(1, p) * betaOld.rows(1, p), n, tau);
  arma::vec res = Y - Z * betaOld;
  double lossOld = sqLossUnif(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerUnif(-res / h, n, tau) / n;
  arma::vec betaNew = betaOld - c0 * gradOld;
  res = Y - Z * betaNew;
  double lossNew = sqLossUnif(res, n, tau, h);
  arma::vec gradNew, gradDiff, betaDiff;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaNew - betaOld, "inf") > tol && ite <= iteMax) {
    gradNew = Z.t() * sqDerUnif(-res / h, n, tau) / n;
    gradDiff = gradNew - gradOld;
    betaDiff = betaNew - betaOld;
    double alpha = c0 / std::pow(ite, std::abs(tau - 0.5));
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), c0) / std::pow(ite, std::abs(tau - 0.5));
    }
    betaOld = betaNew;
    gradOld = gradNew;
    lossOld = lossNew;
    betaNew -= alpha * gradNew;
    res += alpha * Z * gradNew;
    lossNew = sqLossUnif(res, n, tau, h);
    ite++;
  }
  betaNew.rows(1, p) /= arma::stddev(X, 0, 0).t();
  betaNew(0) -= arma::as_scalar(arma::mean(X, 0) * betaNew.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaNew, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
Rcpp::List smqrPara(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, 
                    const double constTau = 1.345, const double tol = 0.00001, const int iteMax = 500) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::sqrt(tau * (1 - tau)) * std::cbrt((double)(p + 1) / n);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  double c0 = 4 * tau * (1 - tau);
  arma::vec betaOld = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  betaOld(0) = quant(Y - Z.cols(1, p) * betaOld.rows(1, p), n, tau);
  arma::vec res = Y - Z * betaOld;
  double lossOld = sqLossPara(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerPara(-res / h, n, tau) / n;
  arma::vec betaNew = betaOld - c0 * gradOld;
  res = Y - Z * betaNew;
  double lossNew = sqLossPara(res, n, tau, h);
  arma::vec gradNew, gradDiff, betaDiff;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaNew - betaOld, "inf") > tol && ite <= iteMax) {
    gradNew = Z.t() * sqDerPara(-res / h, n, tau) / n;
    gradDiff = gradNew - gradOld;
    betaDiff = betaNew - betaOld;
    double alpha = c0 / std::pow(ite, std::abs(tau - 0.5));
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), c0) / std::pow(ite, std::abs(tau - 0.5));
    }
    betaOld = betaNew;
    gradOld = gradNew;
    lossOld = lossNew;
    betaNew -= alpha * gradNew;
    res += alpha * Z * gradNew;
    lossNew = sqLossPara(res, n, tau, h);
    ite++;
  }
  betaNew.rows(1, p) /= arma::stddev(X, 0, 0).t();
  betaNew(0) -= arma::as_scalar(arma::mean(X, 0) * betaNew.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaNew, Rcpp::Named("ite") = ite);
}

// [[Rcpp::export]]
Rcpp::List smqrTrian(const arma::mat& X, const arma::vec& Y, const double tau = 0.5, 
                     const double constTau = 1.345, const double tol = 0.00001, const int iteMax = 500) {
  int n = X.n_rows;
  int p = X.n_cols;
  double h = std::sqrt(tau * (1 - tau)) * std::cbrt((double)(p + 1) / n);
  arma::mat Z(n, p + 1);
  Z.cols(1, p) = standardize(X, p);
  Z.col(0) = arma::ones(n);
  double c0 = 4 * tau * (1 - tau);
  arma::vec betaOld = huberReg(Z, Y, n, p, tol, constTau, iteMax);
  betaOld(0) = quant(Y - Z.cols(1, p) * betaOld.rows(1, p), n, tau);
  arma::vec res = Y - Z * betaOld;
  double lossOld = sqLossTrian(res, n, tau, h);
  arma::vec gradOld = Z.t() * sqDerTrian(-res / h, n, tau) / n;
  arma::vec betaNew = betaOld - c0 * gradOld;
  res = Y - Z * betaNew;
  double lossNew = sqLossTrian(res, n, tau, h);
  arma::vec gradNew, gradDiff, betaDiff;
  int ite = 1;
  while (std::abs(lossNew - lossOld) > tol && arma::norm(betaNew - betaOld, "inf") > tol && ite <= iteMax) {
    gradNew = Z.t() * sqDerTrian(-res / h, n, tau) / n;
    gradDiff = gradNew - gradOld;
    betaDiff = betaNew - betaOld;
    double alpha = c0 / std::pow(ite, std::abs(tau - 0.5));
    double cross = arma::as_scalar(betaDiff.t() * gradDiff);
    if (cross > 0) {
      double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
      double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
      alpha = std::min(std::min(a1, a2), c0) / std::pow(ite, std::abs(tau - 0.5));
    }
    betaOld = betaNew;
    gradOld = gradNew;
    lossOld = lossNew;
    betaNew -= alpha * gradNew;
    res += alpha * Z * gradNew;
    lossNew = sqLossTrian(res, n, tau, h);
    ite++;
  }
  betaNew.rows(1, p) /= arma::stddev(X, 0, 0).t();
  betaNew(0) -= arma::as_scalar(arma::mean(X, 0) * betaNew.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = betaNew, Rcpp::Named("ite") = ite);
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
  arma::vec rst = arma::zeros(p + 1);
  const double a = 3.7;
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    if (abBeta <= lambda) {
      rst(i) = lambda;
    } else if (abBeta <= a * lambda) {
      rst(i) = (a * lambda - abBeta) / (a - 1);
    }
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec cmptLambdaMCP(const arma::vec& beta, const double lambda, const int p) {
  arma::vec rst = arma::zeros(p + 1);
  const double a = 3;
  for (int i = 1; i <= p; i++) {
    double abBeta = std::abs(beta(i));
    if (abBeta <= a * lambda) {
      rst(i) = lambda - abBeta / a;
    }
  }
  return rst;
}

// [[Rcpp::export]]
double l2Loss(const arma::vec& x) {
  return arma::mean(arma::square(x)) / 2;
}
  
// [[Rcpp::export]]
arma::vec gradL2(const arma::mat& X, const arma::vec& Y, const arma::vec& beta, const bool intercept,
                 const int n) {
  arma::vec res = Y - X * beta;
  arma::vec rst = -X.t() * res / n;
  if (!intercept) {
    rst(0) = 0;
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec gradHuber(const arma::mat& X, const arma::vec& Y, const arma::vec& beta, const double tau,
                    const bool intercept, const int n) {
  arma::vec res = Y - X * beta;
  arma::vec rst = X.t() * huberDer(res, n, tau) / n;
  if (!intercept) {
    rst(0) = 0;
  }
  return rst;
}

// [[Rcpp::export]]
arma::vec gradSmq(const arma::mat& X, const arma::vec& Y, const arma::vec& beta, const double tau,
                  const double h, const bool intercept, const int n) {
  arma::vec res = Y - X * beta;
  arma::vec rst = X.t() * sqDerGauss(-res / h, tau) / n;
  if (!intercept) {
    rst(0) = 0;
  }
  return rst;
}

// [[Rcpp::export]]
Rcpp::List lammL2(const arma::mat& X, const arma::vec& Y, const arma::vec& Lambda, 
                  const arma::vec& beta, const double phi, const double gamma, const bool intercept, 
                  const int n, const int p) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad = gradL2(X, Y, beta, intercept, n);
  double loss = l2Loss(Y - X * beta);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = l2Loss(Y - X * betaNew);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) 
      + phiNew * arma::as_scalar(diff.t() * diff) / 2;
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phiNew);
}

// [[Rcpp::export]]
Rcpp::List lammHuber(const arma::mat& X, const arma::vec& Y, const arma::vec& Lambda, 
                     const arma::vec& beta, const double phi, const double tau, const double gamma, 
                     const bool intercept, const int n, const int p) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad = gradHuber(X, Y, beta, tau, intercept, n);
  double loss = huberLoss(Y - X * beta, n, tau);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = huberLoss(Y - X * betaNew, n, tau);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) 
      + phiNew * arma::as_scalar(diff.t() * diff) / 2;
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phiNew);
}

// [[Rcpp::export]]
Rcpp::List lammSmq(const arma::mat& X, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta,
                   const double phi, const double tau, const double h, const double gamma, 
                   const bool intercept, const int n, const int p) {
  double phiNew = phi;
  arma::vec betaNew(p + 1);
  arma::vec grad = gradSmq(X, Y, beta, tau, h, intercept, n);
  double loss = sqLossGauss(Y - X * beta, tau, h);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = sqLossGauss(Y - X * betaNew, tau, h);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) 
      + phiNew * arma::as_scalar(diff.t() * diff) / 2;
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phiNew);
}

// [[Rcpp::export]]
arma::vec lasso(const arma::mat& X, const arma::vec& Y, const double lambda, const int n, const int d, 
                const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, 
                const int iteMax = 500, const bool intercept = false) {
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammL2(X, Y, Lambda, beta, phi, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec scad(const arma::mat& X, const arma::vec& Y, const double lambda, const int n, const int d, 
               const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001,
               const double epsilon_t = 0.001, const int iteMax = 500, const bool intercept = false) {
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammL2(X, Y, Lambda, beta, phi, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0 = arma::zeros(d + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, d);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      listLAMM  = lammL2(X, Y, Lambda, beta, phi, gamma, intercept, n, d);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_t) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec mcp(const arma::mat& X, const arma::vec& Y, const double lambda, const int n, const int d, 
              const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001,
              const double epsilon_t = 0.001, const int iteMax = 500, const bool intercept = false) {
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammL2(X, Y, Lambda, beta, phi, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0 = arma::zeros(d + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, d);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      listLAMM  = lammL2(X, Y, Lambda, beta, phi, gamma, intercept, n, d);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_t) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec huberLasso(const arma::mat& X, const arma::vec& Y, const double lambda, const int n,
                     const int d, const double phi0 = 0.001, const double gamma = 1.5, 
                     const double epsilon_c = 0.001, const int iteMax = 500, 
                     const bool intercept = false, double tfConst = 2.5) {
  arma::vec betaLasso = lasso(X, Y, lambda, n, d, phi0, gamma, epsilon_c, iteMax, intercept);
  arma::vec res = Y - X * betaLasso;
  double tau = tfConst * mad(res);
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammHuber(X, Y, Lambda, beta, phi, tau, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
    res = Y - X * beta;
    tau = tfConst * mad(res);
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec huberSCAD(const arma::mat& X, const arma::vec& Y, const double lambda, const int n,
                    const int d, const double phi0 = 0.001, const double gamma = 1.5, 
                    const double epsilon_c = 0.001, const double epsilon_t = 0.001, 
                    const int iteMax = 500, const bool intercept = false, double tfConst = 2.5) {
  arma::vec betaSCAD = scad(X, Y, lambda, n, d, phi0, gamma, epsilon_c, epsilon_t, iteMax, intercept);
  arma::vec res = Y - X * betaSCAD;
  double tau = tfConst * mad(res);
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammHuber(X, Y, Lambda, beta, phi, tau, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
    res = Y - X * beta;
    tau = tfConst * mad(res);
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0 = arma::zeros(d + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, d);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      listLAMM = lammHuber(X, Y, Lambda, beta, phi, tau, gamma, intercept, n, d);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
        break;
      }
      beta = betaNew;
      res = Y - X * beta;
      tau = tfConst * mad(res);
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec huberMCP(const arma::mat& X, const arma::vec& Y, const double lambda, const int n,
                   const int d, const double phi0 = 0.001, const double gamma = 1.5, 
                   const double epsilon_c = 0.001, const double epsilon_t = 0.001, 
                   const int iteMax = 500, const bool intercept = false, double tfConst = 2.5) {
  arma::vec betaMCP = mcp(X, Y, lambda, n, d, phi0, gamma, epsilon_c, epsilon_t, iteMax, intercept);
  arma::vec res = Y - X * betaMCP;
  double tau = tfConst * mad(res);
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammHuber(X, Y, Lambda, beta, phi, tau, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
    res = Y - X * beta;
    tau = tfConst * mad(res);
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0 = arma::zeros(d + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, d);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      listLAMM = lammHuber(X, Y, Lambda, beta, phi, tau, gamma, intercept, n, d);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
        break;
      }
      beta = betaNew;
      res = Y - X * beta;
      tau = tfConst * mad(res);
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
Rcpp::List smqrLasso(arma::mat X, const arma::vec& Y, double lambda = -1, const double tau = 0.5, 
                     const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, 
                     const int iteMax = 500, const bool intercept = false, 
                     const bool itcpIncluded = false, const double tfConst = 2.5) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  const double h = std::cbrt(std::log(d) / n);
  if (lambda <= 0) {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambda = std::exp((long double)(0.7 * std::log((long double)lambdaMax)
                                      + 0.3 * std::log((long double)lambdaMin)));
  }
  arma::vec beta = huberLasso(X, Y, lambda, n, d, phi0, gamma, epsilon_c, iteMax, intercept, tfConst);
  beta(0) = quant(Y - X.cols(1, d) * beta.rows(1, d), n, tau);
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammSmq(X, Y, Lambda, beta, phi, tau, h, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phi, 
                            Rcpp::Named("lambda") = lambda);
}

// [[Rcpp::export]]
Rcpp::List smqrSCAD(arma::mat X, const arma::vec& Y, double lambda = -1, const double tau = 0.5, 
                    const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, 
                    const double epsilon_t = 0.001, const int iteMax = 500, const bool intercept = false, 
                    const bool itcpIncluded = false, const double tfConst = 2.5) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  const double h = std::cbrt(std::log(d) / n);
  if (lambda <= 0) {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambda = std::exp((long double)(0.7 * std::log((long double)lambdaMax)
                                      + 0.3 * std::log((long double)lambdaMin)));
  }
  arma::vec beta = huberSCAD(X, Y, lambda, n, d, phi0, gamma, epsilon_c, epsilon_t, iteMax, 
                             intercept, tfConst);
  beta(0) = quant(Y - X.cols(1, d) * beta.rows(1, d), n, tau);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammSmq(X, Y, Lambda, beta, phi, tau, h, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0 = arma::zeros(d + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, d);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      listLAMM = lammSmq(X, Y, Lambda, beta, phi, tau, h, gamma, intercept, n, d);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
      break;
    }
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phi, 
                            Rcpp::Named("lambda") = lambda, Rcpp::Named("IteTightening") = iteT);
}

// [[Rcpp::export]]
Rcpp::List smqrMCP(arma::mat X, const arma::vec& Y, double lambda = -1, const double tau = 0.5, 
                   const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, 
                   const double epsilon_t = 0.001, const int iteMax = 500, const bool intercept = false, 
                   const bool itcpIncluded = false, const double tfConst = 2.5) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  const double h = std::cbrt(std::log(d) / n);
  if (lambda <= 0) {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambda = std::exp((long double)(0.7 * std::log((long double)lambdaMax)
                                      + 0.3 * std::log((long double)lambdaMin)));
  }
  arma::vec beta = huberMCP(X, Y, lambda, n, d, phi0, gamma, epsilon_c, epsilon_t, iteMax, 
                            intercept, tfConst);
  beta(0) = quant(Y - X.cols(1, d) * beta.rows(1, d), n, tau);
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, d);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite <= iteMax) {
    ite++;
    listLAMM = lammSmq(X, Y, Lambda, beta, phi, tau, h, gamma, intercept, n, d);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 0;
  // Tightening
  arma::vec beta0 = arma::zeros(d + 1);
  while (iteT <= iteMax) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, d);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      listLAMM = lammSmq(X, Y, Lambda, beta, phi, tau, h, gamma, intercept, n, d);
      betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
      phi = listLAMM["phi"];
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon_t) {
      break;
    }
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phi, 
                            Rcpp::Named("lambda") = lambda, Rcpp::Named("IteTightening") = iteT);
}

// [[Rcpp::export]]
arma::uvec getIndex(const int n, const int low, const int up) {
  arma::vec seq = arma::regspace(0, n - 1);
  return arma::find(seq >= low && seq <= up);
}

// [[Rcpp::export]]
arma::uvec getIndexComp(const int n, const int low, const int up) {
  arma::vec seq = arma::regspace(0, n - 1);
  return arma::find(seq < low || seq > up);
}

// [[Rcpp::export]]
Rcpp::List cvSmqrLasso(arma::mat& X, const arma::vec& Y, Rcpp::Nullable<Rcpp::NumericVector> lSeq = R_NilValue, 
                       int nlambda = 50, const double tau = 0.5, const double phi0 = 0.001, 
                       const double gamma = 1.5, const double epsilon_c = 0.001, 
                       const int iteMax = 500, int nfolds = 5, const bool intercept = false, 
                       const bool itcpIncluded = false, const double tfConst = 2.5) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  int size = n / nfolds;
  arma::vec lambdaSeq = arma::vec();
  if (lSeq.isNotNull()) {
    lambdaSeq = Rcpp::as<arma::vec>(lSeq);
    nlambda = lambdaSeq.size();
  } else {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambdaSeq = exp(arma::linspace(std::log((long double)lambdaMin), 
                                   std::log((long double)lambdaMax), nlambda));
  }
  arma::vec YPred(n);
  arma::vec betaHat(d + 1);
  arma::vec mse(nlambda);
  int low, up;
  arma::uvec idx, idxComp;
  Rcpp::List listILAMM;
  for (int i = 0; i < nlambda; i++) {
    for (int j = 0; j < nfolds; j++) {
      low = j * size;
      up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
      idx = getIndex(n, low, up);
      idxComp = getIndexComp(n, low, up);
      listILAMM = smqrLasso(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i), tau, phi0, gamma, 
                            epsilon_c, iteMax, intercept, true, tfConst);
      betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
      YPred.rows(idx) = X.rows(idx) * betaHat;
    }
    mse(i) = arma::norm(Y - YPred, 2);
  }
  arma::uword cvIdx = arma::index_min(mse);
  listILAMM = smqrLasso(X, Y, lambdaSeq(cvIdx), tau, phi0, gamma, epsilon_c, iteMax, intercept, 
                        true, tfConst);
  betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
  return Rcpp::List::create(Rcpp::Named("beta") = betaHat, Rcpp::Named("lambdaSeq") = lambdaSeq, 
                            Rcpp::Named("mse") = mse, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("nfolds") = nfolds);
}

// [[Rcpp::export]]
Rcpp::List cvSmqrSCAD(arma::mat& X, const arma::vec& Y, Rcpp::Nullable<Rcpp::NumericVector> lSeq = R_NilValue, 
                      int nlambda = 50, const double tau = 0.5, const double phi0 = 0.001, 
                      const double gamma = 1.5, const double epsilon_c = 0.001, 
                      const double epsilon_t = 0.001, const int iteMax = 500, int nfolds = 5, 
                      const bool intercept = false, const bool itcpIncluded = false, 
                      const double tfConst = 2.5) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  int size = n / nfolds;
  arma::vec lambdaSeq = arma::vec();
  if (lSeq.isNotNull()) {
    lambdaSeq = Rcpp::as<arma::vec>(lSeq);
    nlambda = lambdaSeq.size();
  } else {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambdaSeq = exp(arma::linspace(std::log((long double)lambdaMin), 
                                   std::log((long double)lambdaMax), nlambda));
  }
  arma::vec YPred(n);
  arma::vec betaHat(d + 1);
  arma::vec mse(nlambda);
  int low, up;
  arma::uvec idx, idxComp;
  Rcpp::List listILAMM;
  for (int i = 0; i < nlambda; i++) {
    for (int j = 0; j < nfolds; j++) {
      low = j * size;
      up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
      idx = getIndex(n, low, up);
      idxComp = getIndexComp(n, low, up);
      listILAMM = smqrSCAD(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i), tau, phi0, gamma, 
                           epsilon_c, epsilon_t, iteMax, intercept, true, tfConst);
      betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
      YPred.rows(idx) = X.rows(idx) * betaHat;
    }
    mse(i) = arma::norm(Y - YPred, 2);
  }
  arma::uword cvIdx = arma::index_min(mse);
  listILAMM = smqrSCAD(X, Y, lambdaSeq(cvIdx), tau, phi0, gamma, epsilon_c, epsilon_t, iteMax, 
                       intercept, true, tfConst);
  betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
  return Rcpp::List::create(Rcpp::Named("beta") = betaHat, Rcpp::Named("lambdaSeq") = lambdaSeq, 
                            Rcpp::Named("mse") = mse, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("nfolds") = nfolds);
}

// [[Rcpp::export]]
Rcpp::List cvSmqrMCP(arma::mat& X, const arma::vec& Y, Rcpp::Nullable<Rcpp::NumericVector> lSeq = R_NilValue, 
                     int nlambda = 50, const double tau = 0.5, const double phi0 = 0.001, 
                     const double gamma = 1.5, const double epsilon_c = 0.001, 
                     const double epsilon_t = 0.001, const int iteMax = 500, int nfolds = 5, 
                     const bool intercept = false, const bool itcpIncluded = false, 
                     const double tfConst = 2.5) {
  if (!itcpIncluded) {
    arma::mat XX(X.n_rows, X.n_cols + 1);
    XX.cols(1, X.n_cols) = X;
    XX.col(0) = arma::ones(X.n_rows);
    X = XX;
  }
  int n = Y.size();
  int d = X.n_cols - 1;
  int size = n / nfolds;
  arma::vec lambdaSeq = arma::vec();
  if (lSeq.isNotNull()) {
    lambdaSeq = Rcpp::as<arma::vec>(lSeq);
    nlambda = lambdaSeq.size();
  } else {
    double lambdaMax = arma::max(arma::abs(Y.t() * X)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambdaSeq = exp(arma::linspace(std::log((long double)lambdaMin), 
                                   std::log((long double)lambdaMax), nlambda));
  }
  arma::vec YPred(n);
  arma::vec betaHat(d + 1);
  arma::vec mse(nlambda);
  int low, up;
  arma::uvec idx, idxComp;
  Rcpp::List listILAMM;
  for (int i = 0; i < nlambda; i++) {
    for (int j = 0; j < nfolds; j++) {
      low = j * size;
      up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
      idx = getIndex(n, low, up);
      idxComp = getIndexComp(n, low, up);
      listILAMM = smqrMCP(X.rows(idxComp), Y.rows(idxComp), lambdaSeq(i), tau, phi0, gamma, 
                          epsilon_c, epsilon_t, iteMax, intercept, true, tfConst);
      betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
      YPred.rows(idx) = X.rows(idx) * betaHat;
    }
    mse(i) = arma::norm(Y - YPred, 2);
  }
  arma::uword cvIdx = arma::index_min(mse);
  listILAMM = smqrMCP(X, Y, lambdaSeq(cvIdx), tau, phi0, gamma, epsilon_c, epsilon_t, iteMax, 
                      intercept, true, tfConst);
  betaHat = Rcpp::as<arma::vec>(listILAMM["beta"]);
  return Rcpp::List::create(Rcpp::Named("beta") = betaHat, Rcpp::Named("lambdaSeq") = lambdaSeq, 
                            Rcpp::Named("mse") = mse, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("nfolds") = nfolds);
}

