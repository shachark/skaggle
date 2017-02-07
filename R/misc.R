# General utilities I often find useful

# TODO: clean this up... it has become very cluttered

# Preprocessing
# ==================================================================================================

# Find groups of correlated features (based on caret's findCorrelation)
find.correlated.groups = function(x, cutoff = 0.99) {
  averageCorr = colMeans(abs(x))
  averageCorr = as.numeric(as.factor(averageCorr))
  x[lower.tri(x, diag = T)] = NA
  combsAboveCutoff = which(abs(x) > cutoff)

  colsToCheck = ceiling(combsAboveCutoff / nrow(x))
  rowsToCheck = combsAboveCutoff %% nrow(x)

  colsToDiscard = averageCorr[colsToCheck] > averageCorr[rowsToCheck]
  rowsToDiscard = !colsToDiscard

  colsFlagged = pmin(ifelse(colsToDiscard, colsToCheck, NA), ifelse(rowsToDiscard, rowsToCheck, NA), na.rm = T)
  xg = 1:nrow(x)
  for (i in 1:length(rowsToCheck)) {
    xl = min(rowsToCheck[i], colsToCheck[i])
    xh = max(rowsToCheck[i], colsToCheck[i])
    xg[xh] = xg[xl]
  }

  xg = as.numeric(as.factor(xg))
  g = which(table(xg) > 1)
  merge.list = lapply(as.list(g), function(j) which(xg == j))
  names(merge.list) = paste0('GROUP.', unlist(lapply(merge.list, function(j) colnames(x)[j[1]])))
  return (merge.list)
}

# Encode a character or factor with its hexavegisimal value
hexv.encode = function(x, xnew = x) {
  lvls = as.character(unique(x))
  lvls = lvls[order(nchar(lvls), tolower(lvls))]
  return (as.integer(factor(xnew, levels = lvls)))
}

# Encode a feature value with its frequency in the entire dataset
freq.encode = function(x, xnew = x) {
  if (is.factor(x) || is.character(x)) {
    return (as.numeric(factor(xnew, levels = names(sort(table(x))))))
  } else {
    return (approxfun(density(x[!is.na(x)], n = length(x) / 100))(xnew))
  }
}

# Generate CV predictions (aka stacked meta features) from a saturated marginal model.
# NOTE: y is assumed to be NA for test samples

# K-fold version

sms.encode = function (x, y, cv.folds) {
  x = as.factor(x)
  train.idx = !is.na(y)
  y.train = y[train.idx] # should be numeric for lm and ordered for polr
  x.train = x[train.idx]
  x.test = x[!train.idx]

  x.train.enc = rep(NA_real_, length(y.train))
  for (i in 1:length(cv.folds)) {
    idx = cv.folds[[i]]
    x.train.enc[idx] = mean(y.train[-idx])
    bad.idx = which(!(x.train[idx] %in% unique(x.train[-idx])))
    if (length(bad.idx) > 0) {
      idx2 = idx[-bad.idx]
    } else {
      idx2 = idx
    }
    x.train.enc[idx2] = predict(lm(y ~ x, data.frame(y = y.train[-idx], x = x.train[-idx])), data.frame(x = x.train[idx2]))
  }

  bad.idx = which(!(x.test %in% unique(x.train)))
  if (length(bad.idx) > 0) {
    x.test.enc = rep(mean(y.train), length(x.test))
    x.test.enc[-bad.idx] = predict(lm(y ~ x, data.frame(y = y.train, x = x.train)), data.frame(x = x.test[-bad.idx]))
  } else {
    x.test.enc = predict(lm(y ~ x, data.frame(y = y.train, x = x.train)), data.frame(x = x.test))
  }

  x.enc = rep(NA_real_, length(y))
  x.enc[ train.idx] = x.train.enc
  x.enc[!train.idx] = x.test.enc
  return (x.enc)
}

# Leave one out version

# Hmm... consider the binary x, binary y case. If x is left in the data, in addition to this
# encoding of x, then all y values can be exactly recovered from this pair of features! This kind of
# lead also happens to some extent in the general SMS case, and I suppose in stacking in general.
# To keep this in check:
# - Don't use LOO
# - Keep the complexity of the saturated model within reason (don't allow categories that are too
#   sparse, and/or use ridge-like regularization?
# - Add noise - this sounds counterproductive, unhelpful (zero mean noise) and how do you choose the variance?

sms.loo.encode = function (x, y) {
  x = as.character(as.factor(x))
  x[is.na(x)] = 'NA' # FIXME assuming this is not already a factor level...
  x = as.factor(x)
  train.idx = !is.na(y)
  y.train = y[train.idx]
  x.train = x[train.idx]
  x.test = x[!train.idx]

  x.train.enc = rep(NA_real_, length(x.train))
  x.test.enc = rep(NA_real_, length(x.test))

  lvls = levels(x)
  for (l in lvls) {
    idx.lvl = (x.train == l)
    nr.lvl = sum(idx.lvl)
    if (nr.lvl > 1) {
      x.train.enc[idx.lvl] = (sum(y.train[idx.lvl]) - y.train[idx.lvl]) / (nr.lvl - 1)
      #x.train.enc[idx.lvl] = mean(y.train[idx.lvl]) # this leak seems to be better than the alternative
      x.test.enc[x.test == l] = mean(y.train[idx.lvl])
    }
  }

  x.enc = rep(NA_real_, length(y))
  x.enc[ train.idx] = x.train.enc
  x.enc[!train.idx] = x.test.enc
  return (x.enc)
}

# Regularized k-fold version

# This is a saturated marginal regression model y ~ 1 + x, regularized using CV separately for each
# level of x. The regularization is towards the global mean.
rms.train.and.predict = function (x.train, y.train, x.test, min.level.n = 50, nr.folds = 5) {
  if (0) {
    # test code
    x.train = sample(c(1, 2, 3, 4), 1e4, replace = T)
    y.train = as.numeric(runif(1e4) < x.train / 8)
    x.test = sample(c(1, 3, 4, 5), 1e3, replace = T)
    min.level.n = 50 # NOTE: this has to be more than the number of folds
    nr.folds = 5
  }

  # Let's assume this leaks only negligibly
  # FIXME or is it better to predict NA?
  x.train.enc = rep(mean(y.train), length(x.train))
  x.test.enc = rep(mean(y.train), length(x.test))

  lvls = unique(x.train)
  y.train.num = as.numeric(y.train)
  ym = mean(y.train.num)

  nr.alphas = 101
  alphas = seq(0, 1, len = nr.alphas)

  # FIXME I am assuming squared error only
  # FIXME this is an absurdly inefficient implementation, I know
  # It might even be possible to do SURE selection instead of CV

  for (l in lvls) {
    idx.lvl = (x.train == l)
    if (sum(idx.lvl) <= min.level.n) {
      next
    }

    y.train.lvl = y.train.num[idx.lvl]
    if (all(y.train.lvl == y.train.lvl[1])) {
      yhat = y.train.lvl[1]
    } else {
      # Does it make sense to use a different partition per level?
      # I don't see a problem in this context, since as long as the outer folds (in rsms.encode
      # below) are the same for all features we should be within the stacking paradigm (which in
      # itself may be problematic, but my point is that we are not worse off here than usual)
      cv.folds = createFolds(y.train.lvl, k = nr.folds)
      cv.errs = matrix(NA, nr.folds, nr.alphas)

      for (fi in 1:nr.folds) {
        idx = cv.folds[[fi]]
        y.fold.test = y.train.lvl[idx]
        y.fold.train = y.train.lvl[-idx]
        ym.fold = mean(y.fold.train)
        yhats = ym * alphas + ym.fold * (1 - alphas)
        for (ai in 1:nr.alphas) {
          cv.errs[fi, ai] = mean((y.fold.test - yhats[ai]) ^ 2)
        }
      }

      yhat = yhats[which.min(colMeans(cv.errs))]
    }

    x.test.enc[x.test == l] = yhat
  }

  return (x.test.enc)
}

# This uses nested CV to generate predictions from the above model
rsms.encode = function (x, y, cv.folds, min.level.n = 50) {
  x = as.character(as.factor(x))
  x[is.na(x)] = 'NA' # FIXME assuming this is not already a factor level...
  x = as.factor(x)
  train.idx = !is.na(y)
  y.train = y[train.idx] # should be numeric for lm and ordered for polr
  x.train = x[train.idx]
  x.test = x[!train.idx]

  x.train.enc = rep(NA_real_, length(y.train))
  for (i in 1:length(cv.folds)) {
    idx = cv.folds[[i]]
    x.train.enc[idx] = rms.train.and.predict(x.train[-idx], y.train[-idx], x.train[idx], min.level.n, nr.folds = length(cv.folds))
  }

  x.test.enc = rms.train.and.predict(x.train, y.train, x.test, min.level.n, nr.folds = length(cv.folds))

  x.enc = rep(NA_real_, length(y))
  x.enc[ train.idx] = x.train.enc
  x.enc[!train.idx] = x.test.enc
  return (x.enc)
}

# Owen Zhang's 2-way interaction y-encoder
cat2WayAvg = function(data, var1, var2, y, pred0, filter, k, f, lambda = NULL, r_k) {
  # data is a data frame with train and test data
  # var1 and var2 are feature names (column names in <data>) to create an interaction for
  # y is the column name in <data> were the repsonse is
  # pred0 is a default prediction?
  # filter is 1 for train data and 0 for test?

  sub1 = data.frame(v1 = data[, var1], v2 = data[, var2], y = data[, y], pred0 = data[, pred0], filt = filter)
  sum1 = sqldf("SELECT v1, v2, SUM( y ) as sumy, AVG( y ) as avgY, sum(1) as cnt FROM sub1 WHERE filt=1 GROUP BY v1, v2")
  tmp1 = sqldf("SELECT b.v1, b.v2, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub1 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2 ")
  # so now for each row in the original data we also have the matching (v1 X v2) sum y, mean y, and occurance count

  # Fix test examples with no matching train ones
  tmp1$cnt[is.na(tmp1$cnt)] = 0
  tmp1$sumy[is.na(tmp1$sumy)] = 0

  # On the trainset, compute LOO mean y (and on test points, leave it as the full fit)
  tmp1$cnt1 = tmp1$cnt
  tmp1$cnt1[filter] = tmp1$cnt[filter] - 1
  tmp1$sumy1 = tmp1$sumy
  tmp1$sumy1[filter] = tmp1$sumy[filter] - tmp1$y[filter]
  tmp1$avgp = with(tmp1, sumy1/cnt1)

  # Shrink the fit toward the default prediction
  if (!is.null(lambda)) {
    tmp1$lambda = lambda
  } else {
    tmp1$lambda = 1 / (1 + exp((tmp1$cnt1 - k) / f))
  }
  tmp1$adj_avg = (1 - tmp1$lambda) * tmp1$avgp + tmp1$lambda * tmp1$pred0

  # Fill in any gaps with the default prediction
  tmp1$avgp   [is.na(tmp1$avgp   )] = tmp1$pred0[is.na(tmp1$avgp   )]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] = tmp1$pred0[is.na(tmp1$adj_avg)]

  # LOO leaks like shit when used in stacking with some models like trees (although shrinking may
  # improve the situation a little). Add noise to mask this (at the cost of ruining the true signal..)
  tmp1$adj_avg[filter] = tmp1$adj_avg[filter] * (1 + (runif(sum(filter)) - 0.5) * r_k)

  return (tmp1$adj_avg)
}

# Another average-response model, but uses Gilberto's nested CV scheme
# m > 0 does bayesian version (taking a weighted average between level and global response means)

# NOTE: I don't fully buy this... not sure it's necessary or optimal (i.e., not overkill)

.yenc.core = function(x, y, xnew, s0, m) {
  tx = table(x)
  ntx = names(tx)
  s = sapply(ntx, function(x0) sum(y[x == x0]))
  xy = (s + s0) / (tx + m)
  #z = rep(NA_real_, length(xnew))
  #for (i in 1:length(tx)) z[xnew == ntx[i]] = xy[i]
  z = xy[match(xnew, ntx)]
  return (z)
}

yenc.bayes = function(x, y, cv.folds, k, m = 0, xnew = NULL) {
  x = as.character(as.factor(c(x, xnew)))
  x[is.na(x)] = '_*_NA_*_'
  x = as.factor(x)
  xnew = x[(length(y) + 1):length(x)]
  x = x[1:length(y)]
  s0 = mean(y) * m

  if (k == -1) { # testset (use the entire trainset)
    z = .yenc.core(x, y, xnew, s0, m)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      z[idx] = .yenc.core(x[-idx], y[-idx], x[idx], s0, m)
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:length(x))[-idx]
      } else {
        train.idx = (1:length(x))[-c(idx, cv.folds[[k]])]
      }
      z[idx] = .yenc.core(x[train.idx], y[train.idx], x[idx], s0, m)
    }
  }

  return (z)
}

# LMM-based version of yenc (based on Mike Pearmain's code, but uses Gilberto's CV scheme)

.yenc.random.intercept.core = function(x, y, xnew) {
  lmer.fit = lmer(y ~ 1 + (1 | x), data.frame(x = x, y = y), REML = F, verbose = F)
  xlvls = rownames(ranef(lmer.fit)$x)
  x.blup = fixef(lmer.fit) + ranef(lmer.fit)$x[, 1]
  z = x.blup[match(xnew, xlvls)]
}

yenc.random.intercept = function(x, y, cv.folds, k, xnew = NULL) {
  if (k == -1) { # testset (use the entire trainset)
    z = .yenc.random.intercept.core(x, y, xnew)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      z[idx] = .yenc.random.intercept.core(x[-idx], y[-idx], x[idx])
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:length(x))[-idx]
      } else {
        train.idx = (1:length(x))[-c(idx, cv.folds[[k]])]
      }
      z[idx] = .yenc.random.intercept.core(x[train.idx], y[train.idx], x[idx])
    }
  }

  return (z)
}

# GLMM-based version of yenc

.yenc.glmm.core = function(x, y, xnew) {
  if (length(unique(y)) == 2) {
    lmer.fit = glmer(y ~ 1 + (1 | x), data.frame(x = x, y = y), family = binomial)
    xlvls = rownames(ranef(lmer.fit)$x)
    x.blup = 1 / (1 + exp(-(fixef(lmer.fit) + ranef(lmer.fit)$x[, 1])))
    z = x.blup[match(xnew, xlvls)]
  } else {
    lmer.fit = lmer(y ~ 1 + (1 | x), data.frame(x = x, y = y))
    xlvls = rownames(ranef(lmer.fit)$x)
    x.blup = fixef(lmer.fit) + ranef(lmer.fit)$x[, 1]
    z = x.blup[match(xnew, xlvls)]
  }

  return (z)
}

# A simple local smoother
.localsmooth.core = function(x, y, xnew) {
  predict(loess(y ~ x), xnew)
}

yenc.glmm = function(x, y, cv.folds, k, xnew = NULL) {
  if (k == -1) { # testset (use the entire trainset)
    z = .yenc.glmm.core(x, y, xnew)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      z[idx] = .yenc.glmm.core(x[-idx], y[-idx], x[idx])
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:length(x))[-idx]
      } else {
        train.idx = (1:length(x))[-c(idx, cv.folds[[k]])]
      }
      z[idx] = .yenc.glmm.core(x[train.idx], y[train.idx], x[idx])
    }
  }

  return (z)
}

# This version selects automatically between GLM and GLMM

yenc.automatic = function(x, y, cv.folds, k, xnew = NULL, num.nlevels.to.cut = 30, cat.max.nlevels.for.bayes = 30, m = 100) {
  if (is.numeric(x) || is.integer(x)) {
    # Continuous input => nonparametric smooth
    fun = .localsmooth.core
  } else if (nlevels(x) <= cat.max.nlevels.for.bayes) {
    # Not too many levels => use GLM
    # FIXME currently using a Bayes average (equivalent to RMSE ridge regression with hard-wired tuning)
    x = as.character(as.factor(c(x, xnew)))
    x[is.na(x)] = '_*_NA_*_'
    x = as.factor(x)
    xnew = x[(length(y) + 1):length(x)]
    x = x[1:length(y)]
    s0 = mean(y) * m

    fun = function(x, y, xnew) .yenc.core(x, y, xnew, s0, m)
  } else {
    # Too many levels => use GLMM
    fun = .yenc.glmm.core
  }

  if (k == -1) { # testset (use the entire trainset)
    z = fun(x, y, xnew)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      z[idx] = fun(x[-idx], y[-idx], x[idx])
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:length(x))[-idx]
      } else {
        train.idx = (1:length(x))[-c(idx, cv.folds[[k]])]
      }
      z[idx] = fun(x[train.idx], y[train.idx], x[idx])
    }
  }

  return (z)
}

# A template for Gilberto's nested stacking scheme, applied to a univariate model
#
# Typical use case: preprocessing to make usable in XGB features whose association with the response
# is highly nonmonotonic.

smm.generic = function(x, y, fun, cv.folds, k, xnew = NULL, ...) {
  if (k == -1) { # testset (use the entire trainset)
    z = fun(x, y, xnew, ...)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      z[idx] = fun(x[-idx], y[-idx], x[idx], ...)
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:length(x))[-idx]
      } else {
        train.idx = (1:length(x))[-c(idx, cv.folds[[k]])]
      }
      z[idx] = fun(x[train.idx], y[train.idx], x[idx], ...)
    }
  }

  return (z)
}

# This is like above but accepts matrix x/xnew (to still produce vector z)
# FIXME unify the interface
smm.generic2 = function(x, y, fun, cv.folds, k, xnew = NULL, ...) {
  if (k == -1) { # testset (use the entire trainset)
    z = fun(x, y, xnew, ...)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      z[idx] = fun(x[-idx, ], y[-idx], x[idx, ], ...)
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    z = rep(NA_real_, length(y))
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:nrow(x))[-idx]
      } else {
        train.idx = (1:nrow(x))[-c(idx, cv.folds[[k]])]
      }
      z[idx] = fun(x[train.idx, ], y[train.idx], x[idx, ], ...)
    }
  }

  return (z)
}

# This is like above but accepts matrix x/xnew and outputs a matrix
# FIXME unify the interface
smm.generic3 = function(x, y, fun, cv.folds, k, xnew = NULL, ...) {
  if (k == -1) { # testset (use the entire trainset)
    z = fun(x, y, xnew, ...)
  } else if (k == 0) { # full trainset (standard k-fold CV)
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      tmp = fun(x[-idx, ], y[-idx], x[idx, ], ...)
      if (i == 1) {
        z = tmp[rep(1, length(y)), ]
      }
      z[idx, ] = tmp
    }
  } else { # for predicting fold k (like standard CV, but also excluding fold k from any trainset)
    for (i in 1:length(cv.folds)) {
      idx = cv.folds[[i]]
      if (i == k) {
        train.idx = (1:nrow(x))[-idx]
      } else {
        train.idx = (1:nrow(x))[-c(idx, cv.folds[[k]])]
      }
      tmp = fun(x[train.idx, ], y[train.idx], x[idx, ], ...)
      if (i == 1) {
        z = tmp[rep(1, length(y)), ]
      }
      z[idx, ] = tmp
    }
  }

  return (z)
}

# Predicts using frequency encoding per response class
# NOTE: This is suitable for binary/categorical y. The idea is to estimate the Naive Bayes posterior
# feature distribution p(x|y = y0), or something monotinic in it at least.

smm.bayes.core = function(x, y, xnew, y0 = NULL) {
  if (!is.null(y0)) { # (otherwise it is a degenerate mode that does freq.encode)
    x = x[y == y0]
    if (length(x) == 0) return (rep(NA, length(xnew)))
  }
  freq.encode(x, xnew)
}

# Isotonic additive modeling with linear interpolation (e.g., useful for calibration)
# ==================================================================================================

iam.train = function(X, y) {
  if (is.vector(X) || ncol(X) == 1) {
    #return (as.stepfun(isoreg(X, y), right = T))
    return (approxfun(sort(X), isoreg(X, y)$yf, rule = 2))
    #return (splinefun(sort(X), isoreg(X, y)$yf, method = 'monoH.FC'))
  } else {
    return (liso.backfit(X, y))
  }
}

iam.predict = function(fit, new.X) {
  if (is.vector(new.X) || ncol(new.X) == 1) {
    return (fit(new.X))
  } else {
    #return (predict(fit, new.X))
    return (fit & new.X)
  }
}

# Quick correlation check
# ==================================================================================================

corrank = function(X, y) {
  get.cor = function(x) {
    idx = !is.na(x)
    cor(x[idx], y[idx], method = 'spearman')
  }

  View(data.frame(spr = unlist(lapply(X, get.cor))))
}

# XGB prediction with multiple ntreelimit values
# ==================================================================================================

predict.xgb.ltd = function(xgb.fit, newdata, ntreelimits, predleaf = F) {
  preds = NULL
  for (i in 1:length(ntreelimits)) {
    preds = cbind(preds, predict(xgb.fit, newdata, ntreelimit = ntreelimits[i], predleaf = predleaf))
  }
  return (preds)
}

# Misc.
# ==================================================================================================

# The Kaggle user ranking score formula (true to 2016-04 at least)
kaggle.score = function(competition.rank, team.size, nteams, t.elapsed = 0) {
  (100000 / sqrt(team.size)) * (competition.rank ^ -0.75) * (log10(1 + log10(nteams))) * exp(-t.elapsed / 500)
}

# mclapplay for Windows
mclapply.win = function (X, FUN, ..., mc.preschedule = T, mc.set.seed = T, mc.silent = F, mc.cores = 1L, mc.cleanup = T, mc.allow.recursive = T) {
  require(ComputeBackend)
  cores = as.integer(mc.cores)

  # FIXME: I'm ignoring most arguments for now...

  if (cores < 1L) {
    stop("'mc.cores' must be >= 1")
  } else if (cores == 1L) {
    lapply(X, FUN, ...)
  } else {
    config = list()
    config$compute.backend = 'multicore'
    config$nr.cores = mc.cores
    config$package.dependencies = c(sessionInfo()$basePkgs, names(sessionInfo()$otherPkgs))
    config$source.dependencies  = '../utils.r' # this will not work if dependencies change..
    config$cluster.dependencies = NULL
    config$cluster.requirements = NULL
    config$X = as.list(X)
    config$fun = match.fun(FUN)
    config$fun.args = list(...)

    fun.wrapper = function(config, core) {
      idxs = ComputeBackend::compute.backend.balance(length(config$X), config$nr.cores, core)
      nr.idxs = length(idxs)
      if (nr.idxs < 1) return (NULL)
      ret = list()
      for (i in 1:nr.idxs) {
        #cat(date(), 'Core', core, 'Working on idx', idxs[i], '\n')
        args = c(config$X[idxs[i]], config$fun.args)
        names(args)[1] = ''
        ret[[i]] = do.call(config$fun, args)
      }
      return (ret)
    }

    res = compute.backend.run(
      config, fun.wrapper, combine = c,
      package.dependencies = config$package.dependencies,
      source.dependencies  = config$source.dependencies,
      cluster.dependencies = config$cluster.dependencies,
      cluster.requirements = config$cluster.requirements,
      cluster.batch.name = 'dscience'
    )

    names(res) = names(X)
    return (res)
  }
}

if (.Platform$OS.type == 'windows') {
  mclapply = mclapply.win
}
