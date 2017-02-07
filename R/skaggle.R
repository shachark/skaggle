####################################################################################################
#
# General utilities that I've found useful when competing in Kaggle competitions
#
####################################################################################################

# NOTE: Much of this can be cleaned-up, improved, and generalized. It's always work in progress.

# Score functions (aka evaluation metrics)
# ==================================================================================================

#
# Logloss
#

# NOTE: the 1e-15 clamping is how Kaggle usually defines it

# Version for XGB
eval.logloss = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  logloss = -mean(log(ifelse(labels == 0, 1 - preds, preds)))
  return (list(metric = 'logloss', value = logloss))
}

# Version for XGB with multiple limits
eval.logloss.ltd = function(preds, dtrain, ntreelimits) {
  labels = getinfo(dtrain, 'label')
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  logloss = NULL
  for (i in 1:ncol(preds)) {
    logloss = c(logloss, -mean(log(ifelse(labels == 0, 1 - preds[, i], preds[, i]))))
  }
  names(logloss) = ntreelimits
  return (list(metric = 'logloss', value = logloss))
}

# Version for other models (single prediction)
eval.logloss.core = function(preds, labels) {
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  -mean(log(ifelse(labels == 0, 1 - preds, preds)))
}

# Version for other models (multiple predictions)
eval.logloss.core.multi = function(preds, labels) {
  preds = pmax(pmin(preds, 1 - 1e-15), 1e-15)
  logloss = NULL
  for (i in 1:ncol(preds)) {
    logloss = c(logloss, -mean(log(ifelse(labels == 0, 1 - preds[, i], preds[, i]))))
  }
  return (logloss)
}

#
# Mean absolute error
#

# Version for XGB
eval.mae = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  mae = mean(abs(labels - preds))
  return (list(metric = 'mae', value = mae))
}

# Version for XGB with multiple limits
eval.mae.ltd = function(preds, dtrain, ntreelimits) {
  labels = getinfo(dtrain, 'label')
  mae = NULL
  for (i in 1:ncol(preds)) {
    mae = c(mae, mean(abs(labels - preds[, i])))
  }
  names(mae) = ntreelimits
  return (list(metric = 'mae', value = mae))
}

# Version for other models (single prediction)
eval.mae.core = function(preds, labels) {
  mean(abs(labels - preds))
}

# Version for other models (multiple predictions)
eval.mae.core.multi = function(preds, labels) {
  mae = NULL
  for (i in 1:ncol(preds)) {
    mae = c(mae, mean(abs(labels - preds[, i])))
  }
  return (mae)
}

#
# AUC
#

# Version for XGB
eval.auc = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  auc = as.numeric((ROCR::performance(ROCR::prediction(preds, labels), 'auc'))@y.values)
  return (list(metric = 'auc', value = auc))
}

# Version for XGB with multiple limits
eval.auc.ltd = function(preds, dtrain, ntreelimits) {
  labels = getinfo(dtrain, 'label')
  aucs = NULL
  for (i in 1:ncol(preds)) {
    auc = as.numeric((ROCR::performance(ROCR::prediction(preds[, i], labels), 'auc'))@y.values)
    aucs = c(aucs, auc)
  }
  names(aucs) = ntreelimits
  return (list(metric = 'auc', value = aucs))
}

# Version for other models (single prediction)
eval.auc.core = function(preds, labels) {
  auc = as.numeric((ROCR::performance(ROCR::prediction(preds, labels), 'auc'))@y.values)
}

# Version for other models (multiple predictions)
eval.auc.core.multi = function(preds, labels) {
  aucs = NULL
  for (i in 1:ncol(preds)) {
    auc = as.numeric((ROCR::performance(ROCR::prediction(preds[, i], labels), 'auc'))@y.values)
    aucs = c(aucs, auc)
  }
  return (aucs)
}

#
# MAP@3
#

# NOTE: This is a simplified version that assumes each observation has exaclty 3 ordered unique
# predictions, and one true label. Predictions are stored as a (n x 3) matrix.

# Version for XGB
eval.map3 = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  succ = (preds == labels)
  w = 1 / (1:3)
  map3 = mean(succ %*% w)
  return (list(metric = 'map3', value = map3))
}

# Version for XGB with multiple limits
eval.map3.ltd = function(preds, dtrain, ntreelimits) {
  stop('TODO')
}

# Version for other models (single prediction)
eval.map3.core = function(preds, labels) {
  succ = (preds == labels)
  w = 1 / (1:3)
  map3 = mean(succ %*% w)
}

# Version for other models (multiple predictions)
eval.map3.core.multi = function(preds, labels) {
  stop('TODO')
}

#
# MAP@5
#

# NOTE: This is a simplified version that assumes each observation has exaclty 5 ordered unique
# predictions, and one true label. Predictions are stored as a (n x 5) matrix.

# Version for XGB
eval.map5 = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  succ = (preds == labels)
  w = 1 / (1:5)
  map5 = mean(succ %*% w)
  return (list(metric = 'map5', value = map5))
}

# Version for XGB with multiple limits
eval.map5.ltd = function(preds, dtrain, ntreelimits) {
  stop('TODO')
}

# Version for other models (single prediction)
eval.map5.core = function(preds, labels) {
  succ = (preds == labels)
  w = 1 / (1:5)
  map5 = mean(succ %*% w)
}

# Version for other models (multiple predictions)
eval.map5.core.multi = function(preds, labels) {
  stop('TODO')
}

# Model building
# ==================================================================================================

# FIXME get rid of duplicated code here!

train.xgb = function(config) {
  dtrain = xgb.DMatrix(config$xgb.trainset.filename)
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    dvalid = xgboost::slice(dtrain, vidx)
    dtrain = xgboost::slice(dtrain, (1:nrow(dtrain))[-vidx])
    watchlist = list(train = dtrain, valid = dvalid)
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(dtrain), 0.1 * nrow(dtrain))
    dvalid = xgboost::slice(dtrain, vidx)
    dtrain = xgboost::slice(dtrain, (1:nrow(dtrain))[-vidx])
    watchlist = list(train = dtrain, valid = dvalid)
  } else {
    watchlist = list(train = dtrain)
  }

  if (config$proper.validation) {
    dproper = xgb.DMatrix(config$xgb.validset.filename)
    watchlist = c(watchlist, proper = dproper)
  }

  cat(date(), 'Training xgb model\n')

  ntreelimits = seq(config$xgb.ntreelimits.every, config$xgb.params$nrounds, by = config$xgb.ntreelimits.every) * ifelse(is.null(config$xgb.params$num_parallel_tree), 1, config$xgb.params$num_parallel_tree)
  set.seed(config$rng.seed)

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    # NOTE: assuming a linear dependence between the optimal number of rounds and the sample size...
    # Obviously this is not appropriate "in general" (because nothing is).
    nrounds = round(config$xgb.params$nrounds * (1 + 1 / config$nr.folds))
  } else {
    nrounds = config$xgb.params$nrounds
  }

  xgb.fit = xgb.train(
    params            = config$xgb.params,
    nrounds           = nrounds,
    maximize          = config$xgb.params$maximize,
    data              = dtrain,
    watchlist         = watchlist,
    early_stopping_rounds = config$xgb.early.stop.round,
    print_every_n     = ifelse(is.null(config$xgb.params$print.every.n), min(max(100, 100 * round(ceiling(config$xgb.params$nrounds / 20) / 100)), 250), config$xgb.params$print.every.n),
    nthread           = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )

  if (config$measure.importance) {
    cat(date(), 'Examining importance of features in the single XGB model\n')
    impo = xgb.importance(ancillary$feature.names, model = xgb.fit)
    print(impo[1:50, ])
    save(impo, file = 'xgb-feature-importance.RData')
  }

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    if (is.null(config$xgb.params$booster) || config$xgb.params$booster == 'gbtree') {
      preds = predict.xgb.ltd(xgb.fit, dvalid, ntreelimits)
      score = config$eval.score.ltd(preds, dvalid, ntreelimits)$value
    } else {
      preds = predict(xgb.fit, dvalid)
      score = config$eval.score(preds, dvalid)$value
    }
    cat(date(), 'Validation score:\n')
    print(score)

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  dtest = xgb.DMatrix(config$xgb.testset.filename)
  if (is.null(config$xgb.params$booster) || config$xgb.params$booster == 'gbtree') {
    preds = predict.xgb.ltd(xgb.fit, dtest, ntreelimits)
  } else {
    preds = predict(xgb.fit, dtest)
  }

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '-', config$in.fold, '.RData')
  } else {
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
  }
  cat(date(), 'Saving test preds to', fnm, '\n')
  save(preds, file = fnm)

  if (config$proper.validation) {
    preds = predict.xgb.ltd(xgb.fit, dproper, ntreelimits)
    score = config$eval.score.ltd(preds, dproper, ntreelimits)$value
    cat(date(), 'Proper validation score:\n')
    print(score)
  }
}

train.lgb = function(config) {
  dtrain = lgb.Dataset(config$lgb.trainset.filename)
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    dvalid = slice(dtrain, vidx)
    dtrain = slice(dtrain, (1:nrow(dtrain))[-vidx])
    valids = list(train = dtrain, valid = dvalid)
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(dtrain), 0.1 * nrow(dtrain))
    dvalid = slice(dtrain, vidx)
    dtrain = slice(dtrain, (1:nrow(dtrain))[-vidx])
    valids = list(train = dtrain, valid = dvalid)
  } else {
    valids = list(train = dtrain)
  }

  if (config$proper.validation) {
    dproper = lgb.Dataset(config$xgb.validset.filename)
    valids = c(valids, proper = dproper)
  }

  cat(date(), 'Training light GBM model\n')

  set.seed(config$rng.seed)

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    # NOTE: assuming a linear dependence between the optimal number of rounds and the sample size...
    # Obviously this is not appropriate "in general" (because nothing is).
    num_iterations = round(config$lgb.params$num_iterations * (1 + 1 / config$nr.folds))
  } else {
    num_iterations = config$lgb.params$num_iterations
  }

  lgb.fit = lgb.train(
    params            = config$lgb.params,
    nrounds           = num_iterations,
    data              = dtrain,
    valids            = valids,
    early_stopping_rounds = config$early_stopping_rounds,
    eval_freq         = ifelse(is.null(config$lgb.params$eval_freq), min(max(100, 100 * round(ceiling(config$lgb.params$num_iterations / 20) / 100)), 250), config$lgb.params$eval_freq),
    nthread           = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(lgb.fit, dvalid)
    score = config$eval.score(preds, dvalid)$value
    cat(date(), 'Validation score:\n')
    print(score)

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  dtest = lgb.Dataset(config$lgb.testset.filename)
  preds = predict(lgb.fit, dtest)

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '-', config$in.fold, '.RData')
  } else {
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
  }
  cat(date(), 'Saving test preds to', fnm, '\n')
  save(preds, file = fnm)

  if (config$proper.validation) {
    preds = predict(lgb.fit, dproper)
    score = config$eval.score(preds, dproper)$value
    cat(date(), 'Proper validation score:\n')
    print(score)
  }
}

train.ranger = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename)

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training ranger model\n')

  set.seed(config$rng.seed)

  if (config$target.type == 'numeric') {
    ranger.fit = ranger(
      target ~ ., train,
      num.trees = config$ranger.params$num.trees,
      mtry = config$ranger.params$mtry,
      verbose = T,
      write.forest = T,
      num.threads = ifelse(config$compute.backend != 'serial', 1, config$nr.threads))
  } else if (config$target.type == 'categorical') {
    ranger.fit = ranger(
      as.factor(target) ~ ., train,
      num.trees = config$ranger.params$num.trees,
      mtry = config$ranger.params$mtry,
      verbose = T,
      write.forest = T,
      probability = T,
      num.threads = ifelse(config$compute.backend != 'serial', 1, config$nr.threads))
  }

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(ranger.fit, valid)$predictions
    if (config$target.type == 'categorical') {
      preds = preds[, 2] # well, for the binary case anyway
    }
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(ranger.fit, test)$predictions
    if (config$target.type == 'categorical') {
      preds = preds[, 2] # well, for the binary case anyway
    }
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.best = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training best model\n')

  set.seed(config$rng.seed)

  all.scores = rep(Inf, ncol(train))
  for (i in 1:ncol(train)) {
    if (names(train)[i] != 'target') {
      all.scores[i] = config$eval.score.core(train[[i]], train$target)
    }
  }

  if (config$score.to.maximize) {
    best.tag = names(train)[which.max(all.scores)]
    score = max(all.scores)
  } else {
    best.tag = names(train)[which.min(all.scores)]
    score = min(all.scores)
  }
  cat(date(), ' Best model: ', best.tag, ' (score ', score, ')\n', sep = '')

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = valid[[best.tag]]
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = test[[best.tag]]
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.blend = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  # This model doesn't really need any training

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = apply(valid[, -which(names(valid) == 'target'), drop = F], 1, config$belnd.fun)
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = apply(test, 1, config$belnd.fun)
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.glm = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training glm model\n')

  mdl = glm(target ~ ., train, family = config$glm.family)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(mdl, valid, type = 'response')
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(mdl, test, type = 'response')
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.nnls = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training nnls model\n')

  beta = coef(nnls(data.matrix(train[, -which(names(train) == 'target'), drop = F]), train$target))

  if (config$nnls.normalize) {
    beta = beta / sum(beta)
  }

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = c(data.matrix(valid[, -which(names(valid) == 'target'), drop = F]) %*% beta)
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = data.matrix(test) %*% beta
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.glmnet = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training glmnet model\n')

  set.seed(config$rng.seed)

  # One option is to let is tune lambda using cv.glmnet. Another is to use multiple values and stack later.

  if (0) {
    library(doParallel)
    #registerDoParallel(config$nr.threads)
    registerDoParallel(4)
    cvres = cv.glmnet(data.matrix(train[, -which(names(train) == 'target'), drop = F]), train$target, family = config$glmnet.params$family, lambda = config$glmnet.params$lambda, alpha = config$glmnet.params$alpha, parallel = T)
    stopImplicitCluster()
  }

  mdl = glmnet(data.matrix(train[, -which(names(train) == 'target'), drop = F]), train$target, family = config$glmnet.params$family, alpha = config$glmnet.params$alpha)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = predict(mdl, data.matrix(valid[, -which(names(valid) == 'target'), drop = F]), type = 'response', s = config$glmnet.params$lambda)
    score = config$eval.score.core.multi(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = predict(mdl, data.matrix(test), type = 'response', s = config$glmnet.params$lambda)
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.et = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training extra-trees model\n')

  set.seed(config$rng.seed)

  if (config$target.type == 'categorical') {
    y = as.factor(train$target)
  } else {
    y = train$target
  }

  mdl = extraTrees(
    x = data.matrix(train[, -which(names(train) == 'target'), drop = F]),
    y = y,
    ntree = config$et.params$ntree,
    mtry = config$et.params$mtry,
    nodesize = config$et.params$nodesize,
    numThreads = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    if (config$target.type == 'categorical') {
      # FIXME this is for a binary target
      preds = predict(mdl, data.matrix(valid[, -which(names(valid) == 'target'), drop = F]), probability = T)[, 2]
    } else {
      preds = predict(mdl, data.matrix(valid[, -which(names(valid) == 'target'), drop = F]))
    }
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    if (config$target.type == 'categorical') {
      # FIXME this is for a binary target
      preds = predict(mdl, data.matrix(test), probability = T)[, 2]
    } else {
      preds = predict(mdl, data.matrix(test))
    }
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.rf = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training random forest model\n')

  set.seed(config$rng.seed)

  if (0) {
    # Train this model only on the categorical features
    cat.features = names(train)[which(lapply(train, class) == 'factor')]
    frmla = as.formula(paste('as.factor(target) ~', paste(cat.features, collapse = ' + ')))
  } else {
    frmla = (target ~ .)
  }

  # mdl = randomForest(frmla, train, do.trace = 100,
  #   ntree = config$rf.params$ntree,
  #   mtry = config$rf.params$mtry,
  #   nodesize = config$rf.params$nodesize
  # )

  gc()
  h2o::h2o.init(nthreads = config$nr.threads, max_mem_size = config$h2o.max.mem)
  train$target = as.factor(train$target)
  feature.names = names(train)[names(train) != 'target']
  train.for.h2o = h2o::as.h2o(train)
  mdl = h2o::h2o.randomForest(feature.names, 'target', train.for.h2o, mtries = config$rf.params$mtry, ntrees = config$rf.params$ntree, max_depth = config$rf.params$max.depth)
  # TODO: try binomial_double_trees, tune nbins_cats (leaving factors in the data) ...

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    #preds = predict(mdl, valid, type = 'prob')[, 2]
    valid.for.h2o = h2o::as.h2o(valid)
    preds = as.data.frame(h2o::h2o.predict(mdl, valid.for.h2o))[, 3]

    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    #preds = predict(mdl, test, type = 'prob')[, 2]
    test.for.h2o = h2o::as.h2o(test)
    preds = as.data.frame(h2o::h2o.predict(mdl, test.for.h2o))[, 3]

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }

  h2o.removeAll()
  gc()
}

train.knn = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training knn model\n')
  # (actual training is separate for valid and test)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    knn.res = nn2(train[, -which(names(train) == 'target')], query = valid[, -which(names(valid) == 'target')],  k = config$knn.params$k, eps = config$knn.params$eps)
    knn.preds = matrix(train$target[knn.res$nn.idx], ncol = config$knn.params$k)
    preds = rowMeans(knn.preds)
    #ownn.res = ownn(train[, -which(names(train) == 'target')], valid[, -which(names(valid) == 'target')], cl = train$target, k = config$knn.params$k, prob = T)
    #preds = ifelse(ownn.res$ownnpred == 1, attr(ownn.res$ownnpred, 'prob'), 1 - attr(ownn.res$ownnpred, 'prob'))

    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    knn.res = nn2(train[, -which(names(train) == 'target')], query = test,  k = config$knn.params$k, eps = config$knn.params$eps)
    knn.preds = matrix(train$target[knn.res$nn.idx], ncol = config$knn.params$k)
    preds = rowMeans(knn.preds)
    #ownn.res = ownn(train[, -which(names(train) == 'target')], test, cl = train$target, k = config$knn.params$k, prob = T)
    #preds = ifelse(ownn.res$ownnpred == 1, attr(ownn.res$ownnpred, 'prob'), 1 - attr(ownn.res$ownnpred, 'prob'))

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.iso = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training iso model\n')

  #stopifnot(ncol(train) == 2) # only supports univariate isoreg for now (maybe use liso.. but heavy for this sample size!)
  #isofun = as.stepfun(isoreg(, train$target))
  fit = iam.train(data.matrix(train[, -which(names(train) == 'target')]), train$target)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    #preds = isofun(data.matrix(valid[, -which(names(valid) == 'target')]))
    preds = iam.predict(fit, data.matrix(valid[, -which(names(valid) == 'target')]))
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    #preds = isofun(test)
    preds = iam.predict(fit, data.matrix(test))
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.nnet = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training nnet model\n')

  gc()
  h2o::h2o.init(nthreads = config$nr.threads, max_mem_size = config$h2o.max.mem)
  train$target = as.factor(train$target)
  feature.names = names(train)[names(train) != 'target']
  train.for.h2o = h2o::as.h2o(train)

  mdl = h2o::h2o.deeplearning(
    feature.names, 'target', train.for.h2o,
    activation = 'TanhWithDropout',
    hidden = c(1024, 1024),
    input_dropout_ratio = 0.05,
    hidden_dropout_ratios = c(0.9, 0.9),
    rate = 0.001,
    epochs = 100,
    #l1 = 1e-7,
    #l2 = 1e-7,
    #train_samples_per_iteration = 2000,
    #max_w2 = 10,
    stopping_metric = 'score',
    seed = config$rng.seed
  )

  # Things to try:
  # - Leave factors in the data and let H2O handle them
  # - Other activation functions
  # - Other tolopogies...
  # - Look at logs and try to figure out how many epochs...
  # - Adaptive learning (enable, and tune rho & epsilon...)
  # - Regularization...

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    valid.for.h2o = h2o::as.h2o(valid)
    preds = as.data.frame(h2o::h2o.predict(mdl, valid.for.h2o))$p1

    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    test.for.h2o = h2o::as.h2o(test)
    preds = as.data.frame(h2o::h2o.predict(mdl, test.for.h2o))$p1

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }

  h2o.removeAll()
  gc()
}

train.nb = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training nb model\n')

  gc()
  h2o::h2o.init(nthreads = config$nr.threads, max_mem_size = config$h2o.max.mem)
  train$target = as.factor(train$target)
  feature.names = names(train)[names(train) != 'target']
  train.for.h2o = h2o::as.h2o(train)

  mdl = h2o::h2o.naiveBayes(feature.names, 'target', train.for.h2o)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    valid.for.h2o = h2o::as.h2o(valid)
    preds = as.data.frame(h2o::h2o.predict(mdl, valid.for.h2o))$p1

    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    test.for.h2o = h2o::as.h2o(test)
    preds = as.data.frame(h2o::h2o.predict(mdl, test.for.h2o))$p1

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }

  h2o.removeAll()
  gc()
}

train.h2o.gbm = function(config) {
  load(config$ancillary.filename) # => ancillary

  gc()
  h2o::h2o.init(nthreads = config$nr.threads, max_mem_size = config$h2o.max.mem)

  if (0) {
    # I coulnd't get this to work (don't have a library that can write sparse matrices in svmlight format correctly?)
    train = h2o::h2o.uploadFile(config$xgb.trainset.filename, parse_type = 'SVMLight', col.names = ancillary$feature.names)
  } else {
    load(config$dataset.filename) # => train, test
  }

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  ### for debugging
  #train = head(train, 10000)
  ###

  cat(date(), 'Training H2O gbm model\n')

  mdl = h2o::h2o.gbm(ancillary$feature.names, 'target', h2o::as.h2o(train),
                     distribution             = config$h2o.gbm.params$distribution,
                     stopping_metric          = config$h2o.gbm.params$stopping_metric,
                     ntrees                   = config$h2o.gbm.params$ntrees,
                     learn_rate               = config$h2o.gbm.params$learn_rate,
                     max_depth                = config$h2o.gbm.params$max_depth,
                     min_rows                 = config$h2o.gbm.params$min_rows,
                     sample_rate              = config$h2o.gbm.params$sample_rate,
                     col_sample_rate_per_tree = config$h2o.gbm.params$col_sample_rate_per_tree,
                     stopping_rounds          = config$h2o.gbm.params$stopping_rounds,
                     stopping_tolerance       = config$h2o.gbm.params$stopping_tolerance
  )

  gc()

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = as.data.frame(predict(mdl, h2o::as.h2o(valid)))$predict

    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = as.data.frame(predict(mdl, h2o::as.h2o(test)))$predict

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }

  h2o::h2o.removeAll()
  gc()
}

train.mxnet = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
    eval.data = list(data = as.array(t(valid[, -which(names(train) == 'target'), drop = F])), label = as.array(valid$target))
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
    eval.data = list(data = as.array(t(valid[, -which(names(train) == 'target'), drop = F])), label = as.array(valid$target))
  } else {
    eval.data = NULL
  }

  gc(verbose = F)
  net = config$mxnet.params$define.net(config)

  seed.rep.seeds = sample(1e6, config$mxnet.params$nr.seed.reps)
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    seed.rep.valid.pred = matrix(NA_real_, nrow(valid), config$mxnet.params$nr.seed.reps)
  }
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    seed.rep.test.pred  = matrix(NA_real_, nrow(test ), config$mxnet.params$nr.seed.reps)
  }

  trace.err = T

  if (.Platform$OS.type != 'windows') {
    ctx = lapply(1:floor(config$nr.threads / 2), mx.cpu)
  } else {
    ctx = mx.cpu(1)
  }

  for (seed.rep.i in 1:config$mxnet.params$nr.seed.reps) {
    cat(date(), ' Taining mxnet model ', seed.rep.i, ' of ', config$mxnet.params$nr.seed.reps, ', seed ', seed.rep.seeds[seed.rep.i], '\n', sep = '')

    mx.set.seed(seed.rep.seeds[seed.rep.i])
    #Sys.setenv(MXNET_CPU_WORKER_NTHREADS = 2)

    args = list(
      symbol           = net,
      X                = as.array(t(train[, -which(names(train) == 'target'), drop = F])),
      y                = as.array(train$target),
      ctx              = ctx,
      num.round        = config$mxnet.params$num.round,
      optimizer        = config$mxnet.params$optimizer,
      eval.data        = eval.data,
      eval.metric      = config$mxnet.params$mx.metric,
      array.batch.size = config$mxnet.params$batch.size,
      array.layout     = 'colmajor',
      verbose          = trace.err
    )

    if (config$mxnet.params$optimizer == 'sgd') {
      args$learning.rate = config$mxnet.params$sgd.learning.rate
      args$momentum      = config$mxnet.params$sgd.momentum
      args$wd            = config$mxnet.params$sgd.wd
    } else if (config$mxnet.params$optimizer == 'adam') {
      # TODO make params for this
      args$learning.rate = 0.001
      args$beta1         = 0.9   # Exponential decay rate for the first moment estimates
      args$beta2         = 0.999 # Exponential decay rate for the second moment estimates
      args$epsilon       = 1e-8  # L2 regularization coefficient add to all the weights
      args$wd            = 0
    } else {
      stop('TODO')
    }

    if (trace.err) {
      mxlog = mx.metric.logger$new()
      args$epoch.end.callback = mx.callback.log.train.metric(1, mxlog)
    }

    mdl = do.call(mx.model.FeedForward.create, args)

    if (trace.err) {
      plot(mxlog$train, type = 'l', xlab = 'Epoch', ylab = 'MAE', ylim = c(1100, 1200))
      lines(mxlog$eval, col = 2)
      legend('topright', legend = c('Training', 'Validation'), fill = 1:2)
    }

    if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
      seed.rep.valid.pred[, seed.rep.i] = predict(mdl, eval.data$data, array.layout = 'colmajor')
      preds = rowMeans(seed.rep.valid.pred, na.rm = T)
      score = config$eval.score.core(preds, valid$target)
      cat(date(), 'Validation score:', score, '\n')
    }
    if ((!is.null(config$in.fold) && config$in.fold == -1)) {
      seed.rep.test.pred[, seed.rep.i] = predict(mdl, as.array(t(test)), array.layout = 'colmajor')
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold != -1)) {
    preds = rowMeans(seed.rep.valid.pred, na.rm = T)
    fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
    cat(date(), 'Saving valid preds to', fnm, '\n')
    save(preds, file = fnm)
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = rowMeans(seed.rep.test.pred, na.rm = T)
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.gbm = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training gradient boosted machine model\n')

  set.seed(config$rng.seed)

  if (config$target.type == 'categorical') {
    y = as.factor(train$target)
  } else {
    y = train$target
  }

  if (config$gbm.params$all.var.monotone) {
    var.monotone = rep(1, ncol(train) - 1)
  } else {
    var.monotone = NULL
  }

  mdl = gbm.fit(x = data.matrix(train[, -which(names(train) == 'target'), drop = F]), y = y, keep.data = F,
                distribution      = config$gbm.params$distribution,
                n.trees           = config$gbm.params$n.trees,
                shrinkage         = config$gbm.params$shrinkage,
                interaction.depth = config$gbm.params$interaction.depth,
                bag.fraction      = config$gbm.params$bag.fraction,
                n.minobsinnode    = config$gbm.params$n.minobsinnode
  )

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    if (config$target.type == 'categorical') {
      stop('TODO')
    } else {
      preds = predict(mdl, data.matrix(valid[, -which(names(valid) == 'target'), drop = F]), type = 'response', n.trees = config$gbm.params$pred.n.trees)
    }
    score = config$eval.score.core.multi(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    if (config$target.type == 'categorical') {
      stop('TODO')
    } else {
      preds = predict(mdl, data.matrix(test), type = 'response')
    }
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.rgf = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  if (config$target.type == 'categorical') {
    y = (as.numeric(train$target) - 1.5) * 2 # RGF expects labels of classes to be in {-1,1}
  } else {
    y = train$target
  }

  cat(date(), 'Training regularized greedy forest model\n')
  warning('I did not get a chance to test this wrapper yet')

  # NOTE: RGF source is from: http://tongzhang-ml.org/software/rgf/index.html

  set.seed(config$rng.seed)

  train.x.fn = paste0(config$tmp.dir, '/train-x.rgfd')
  train.y.fn = paste0(config$tmp.dir, '/train-y.rgfd')
  # NOTE: RGF also supports an input weight file
  eval.x.fn  = paste0(config$tmp.dir, '/eval-x.rgfd')
  model.fn   = paste0(config$tmp.dir, '/model.rgf')
  preds.fn   = paste0(config$tmp.dir, '/preds.rgfd')

  # NOTE: RGF also accepts input in sparse format. See the guide PDF that comes with the package.
  fwrite(train[, -which(names(train) == 'target'), drop = F], train.x.fn, sep = ' ', col.names = F)
  fwrite(y                                                  , train.x.fn, sep = ' ', col.names = F)

  if (.Platform$OS.type == 'windows') {
    egf.exec = '../../Tools/rgf1.2/bin/rgf.exe.file'
  } else {
    egf.exec = 'rgf'
  }

  cmd.rgf = paste0(egf.exec, ' train train_x_fn=', train.x.fn, ',train_y_fn=', train.y.fn, ',model_fn_prefix=', model.fn,
    ',algorithm=', config$rgf.params$algorithm, ',NormalizeTarget',
    ',loss=', config$rgf.params$loss, ',test_interval', config$rgf.params$max_leaf_forest,
    ',reg_L2=', config$rgf.params$reg_L2, ',reg_sL2=', config$rgf.params$reg_sL2,
    ',max_leaf_forest=', config$rgf.params$max_leaf_forest, ',min_pop=', config$rgf.params$min_pop
  )

  system(cmd.rgf)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    fwrite(valid[, -which(names(valid) == 'target'), drop = F], eval.x.fn, sep = ' ', col.names = F)
    cmd.rgf = paste0('egf.exec predict test_x_fn=', eval.x.fn, ',prediction_fn=', preds.fn, ',model_fn=', model.fn, '-01') # FIXME look at multiple nr trees like in xgb function
    system(cmd.rgf)
    preds = fread(preds.fn)$V1
    if (config$target.type == 'categorical') {
      preds = (preds + 1) / 2
      preds = ifelse(preds > 1, 1, preds)
      preds = ifelse(preds < 0, 0, preds)
    }

    score = config$eval.score.core.multi(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    fwrite(test, eval.x.fn, sep = ' ', col.names = F)
    cmd.rgf = paste0('egf.exec predict test_x_fn=', eval.x.fn, ',prediction_fn=', preds.fn, ',model_fn=', model.fn, '-01')
    system(cmd.rgf)
    preds = fread(preds.fn)$V1
    if (config$target.type == 'categorical') {
      preds = (preds + 1) / 2
      preds = ifelse(preds > 1, 1, preds)
      preds = ifelse(preds < 0, 0, preds)
    }

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.vw = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training Vowpal Wabbit model\n')
  warning('I did not get a chance to test this wrapper yet')

  # NOTE: VW can be found https://github.com/JohnLangford/vowpal_wabbit

  set.seed(config$rng.seed)

  if (config$target.type == 'categorical') {
    y = (as.numeric(train$target) - 1.5) * 2 # VW expects labels of classes to be in {-1,1}
  } else {
    y = train$target
  }

  # FIXME I imagine this can be done much faster with data.table like I've done in FFM
  # Also the code is very bloated and could be trimmed down considerably
  # The VW distribution includes an R package (r.vw) - can't I use that?
  write.vw.input = function(x, y, tag = NULL, weights = NULL, namespaces = NULL, fileName) {
    def_scipen = options('scipen'); options(scipen = 999)
    vx = c()

    name_spaces = rep('F', ncol(x))
    if (!is.null(namespaces) && length(namespaces) != ncol(x)) stop('Namespaces must be a character vector of same length as number of columns')
    if (!is.null(namespaces)) name_spaces = namespaces

    out_df = data.frame(x, stringsAsFactors = F)

    nm = names(x)
    prev_nm_space = ''

    for (i in 1:ncol(x)) {
      nm_space = name_spaces[i]
      vx = x[,i]

      if (!(is.numeric(vx) || is.integer(vx))) stop(paste(nm[i], 'Non-numerical input not supported'))

      if (prev_nm_space == '' || prev_nm_space != nm_space) {
        vx = paste(paste('|', nm_space, ' ', i, sep = ''), vx, sep = ':');
      } else {
        vx = paste(i, vx, sep = ':')
      }

      out_df[,i] = vx
      prev_nm_space = nm_space
      cat('.')

    }

    vx_tag = tag
    if (missing(tag) || is.null(tag)) {
      vx_tag = rep(1, nrow(x))
    }
    vx_weight = weights
    if (missing(weights) || is.null(weights)) {
      vx_weight  = rep(1, nrow(x))
    }
    vx_target = y
    if (missing(y) || is.null(y)) {
      vx_target = rep(0, nrow(x))
    }

    vx_meta = paste0(vx_target, vx_weight, paste(vx_tag,out_df[, 1]))

    out_df = cbind(meta = vx_meta, out_df[, 2:ncol(out_df)])

    if (missing(fileName)) {
      options(scipen = def_scipen)
      return (out_df)
    } else {
      fwrite(out_df, fileName, sep = ' ', quote = F, col.names = F)
      options(scipen = def_scipen)
      return (NULL)
    }
  }

  if (.Platform$OS.type == 'windows') {
    # kill vw.exe process (sometimes it seems to become a zombie process)
    suppressWarnings(system2('taskkill', '/IM vw.exe /f', stdout = NULL, stderr = NULL, wait = T))
    vw.exec = '../../Tools/VW/vw.exe'
  } else {
    vw.exec = 'vw'
  }

  train.fn = paste0(config$tmp.dir, '/train.vwd')
  eval.fn  = paste0(config$tmp.dir, '/eval.vwd')
  model.fn = paste0(config$tmp.dir, '/model.vw')
  preds.fn = paste0(config$tmp.dir, '/preds.vwd')

  vx_params = config$vw.params
  vx_params$data = train.fn
  vx_params$final_regressor = model.fn

  train.cmd_line = ''
  for(p in names(vx_params)) {
    train.cmd_line = paste(train.cmd_line, paste0('--', p, ' ', vx_params[p]))
  }

  pr_params = config$vw.params
  pr_params$initial_regressor = model.fn
  pr_params$predictions = preds.fn
  pr_params$testonly = ''
  pr_params$data = eval.fn

  pred.cmd_line = ''
  for(p in names(pr_params)) {
    pred.cmd_line = paste0(pred.cmd_line, paste0('--', p, ' ', pr_params[p]))
  }

  write.vw.input(data.matrix(train[, -which(names(train) == 'target'), drop = F]), y, train.fn) # TODO: use tag, weights, namespaces
  system2(vw.exec, train.cmd_line, args = train.cmd_line, stdout = ifelse(do.trace, '', NULL), stderr = '', wait = T)

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    write.vw.input(data.matrix(valid[, -which(names(train) == 'target'), drop = F]), rep(1, nrow(valid)), eval.fn)
    if (file.exists(pr_params$predictions)) file.remove(pr_params$predictions)
    system2(vw.exec, pred.cmd_line, args = pred.cmd_line, stdout = ifelse(do.trace, '', NULL), stderr = '', wait = T)
    preds = fread(preds.fn, header = F, sep = ' ')$V1

    score = config$eval.score.core.multi(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    write.vw.input(data.matrix(test), rep(1, nrow(test)), eval.fn)
    if (file.exists(pr_params$predictions)) file.remove(pr_params$predictions)
    system2(vw.exec, pred.cmd_line, args = pred.cmd_line, stdout = ifelse(do.trace, '', NULL), stderr = '', wait = T)
    preds = fread(preds.fn, header = F, sep = ' ')$V1

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

train.ffm = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary

  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }

  cat(date(), 'Training Field-aware Factorization Machine model\n')
  warning('I did not get a chance to test this wrapper yet; See a customized version in my Outbrain comp code')

  # NOTE: FFM source is from: https://github.com/guestwalk/libffm
  # FIXME: There are newer implementations out there that I could try (e.g., see Outbrain comp forum)

  set.seed(config$rng.seed)

  dt.to.libffm = function(dat, labels) {
    # The input file format for libFFM is
    # label field1:index1:value1 field2:index2:value2 ...
    # where label is either 0 or 1, and the remaining is a variable length list of nonzero feature
    # identifiers and their values. "Field"s are indexes (starting from 0) of feature groups (for OHE:
    # groups identify the original feature that was OHE), "index"es are feature identifiers within
    # groups (for OHE: category number in OHE order, starting from 0), and "value"s are... values
    # (for OHE: always 1).

    fld.offset = -1
    idx.offset = 0
    ttl.idx = 0

    ffm.encode.cat = function(x, newfld) {
      if (newfld) {
        fld.offset <<- fld.offset + 1
        idx.offset <<- 0
      }
      isnax = is.na(x)
      index = as.integer(x) - 1
      index[isnax] = nlevels(x)
      encx = paste(fld.offset, idx.offset + index, 1, sep = ':')
      idx.delta = nlevels(x) + any(isnax)
      idx.offset <<- idx.offset + idx.delta
      ttl.idx <<- ttl.idx + idx.delta
      return (encx)
    }

    ffm.encode.qua = function(x, newfld) {
      if (newfld) {
        fld.offset <<- fld.offset + 1
        idx.offset <<- 0
      }
      isnax = is.na(x)
      index = ifelse(isnax, idx.offset + 1, idx.offset)
      value = ifelse(isnax, 1, round(x, digits = 4)) # FIXME does digits matter? how many are enough, how many are supported?
      encx = paste(fld.offset, index, value, sep = ':')
      idx.delta = 1 + any(isnax)
      idx.offset <<- idx.offset + idx.delta
      ttl.idx <<- ttl.idx + idx.delta
      return (encx)
    }

    ffm.encode.qua.bin = function(x, newfld) {
      # Bin to some manageable number of categories
      m = uniqueN(x)
      mb = ifelse(m < 120, m, min(120, floor(m / 6)))
      ffm.encode.cat(cut2(x, g = mb), newfld)
    }

    ffm.encode.qua.cat = function(x, newfld) {
      # acutally treat as categorical (this works if the quantiative feature only takes a handful of unique values)
      ffm.encode.cat(as.factor(x), newfld)
    }

    labels[is.na(labels)] = 0
    dffm = data.table(label = labels)

    # FIXME: this is generic and won't work for every problem. Can add sophisticated parameterization.
    for (col in feature.names) {
      if (class(dat[[col]])[1] %in% c('integer', 'numeric')) {
        dffm[, (col) := ffm.encode.qua(dat[[col]], T)]
      } else if (class()[1] == 'factor') {
        dffm[, (col) := ffm.encode.cat(dat[[col]], T)]
      }
    }

    #cat(date(), 'Total fields:', fld.offset, ', total indexes:', ttl.idx, '\n')

    return (dffm)
  }

  # NOTE: for the coding to be consistent, this has to be done on the joint data. That's awkward
  # and inefficient in the generic pipline. Should consider major refactoring, but tbh the bigger
  # datasets can't go through the generic pipeline anyway and need customized pipes.

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    dat = rbind(train[, -which(names(train) == 'target'), drop = F], valid[, -which(names(train) == 'target'), drop = F], test)
    labels = c(train$target, valid$targets, rep(0, nrow(test)))
  } else {
    dat = rbind(train[, -which(names(train) == 'target'), drop = F], test)
    labels = c(train$target, rep(0, nrow(test)))
  }

  dffm = dt.to.libffm(dat, labels)

  ffm.train.fn     = paste0(config$tmp.dir, '/train.ffm')
  ffm.valid.fn     = paste0(config$tmp.dir, '/valid.ffm')
  ffm.test.fn      = paste0(config$tmp.dir, '/test.ffm' )
  ffm.model.fn     = paste0(config$tmp.dir, '/ffm.model')
  ffm.preds.fn     = paste0(config$tmp.dir, '/ffm.preds')

  fwrite(dffm[1:nrow(train)], ffm.train.fn, sep = ' ', col.names = F)
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    fwrite(dffm[nrow(train) + (1:nrow(valid))], ffm.valid.fn, sep = ' ', col.names = F)
    fwrite(dffm[nrow(train) + nrow(valid) + (1:nrow(test))], ffm.test.fn, sep = ' ', col.names = F)
  } else {
    fwrite(dffm[nrow(train) + (1:nrow(test))], ffm.test.fn, sep = ' ', col.names = F)
  }

  rm(dffm); gc()

  if (.Platform$OS.type == 'windows') {
    ffm.train.exec = '../../Tools/libffm/windows/ffm-train.exe'
    ffm.pred.exec = '../../Tools/libffm/windows/ffm-predict.exe'
  } else {
    ffm.train.exec = './ffm-train'
    ffm.pred.exec = './ffm-predict'
  }

  if (0) {
    # TODO: for early stopping on a validation set:
    system(paste(ffm.train.exec, '-l', config$ffm.params$ffm.l, '-k', config$ffm.params$ffm.k, '-r', config$ffm.params$ffm.r, '-t', config$ffm.params$ffm.t.max, '-s', config$ffm.params$ffm.s, '-p', ffm.valid.fn, '--auto-stop ', ffm.train.fn, ffm.model.fn))
  } else {
    system(paste(ffm.train.exec, '-l', config$ffm.params$ffm.l, '-k', config$ffm.params$ffm.k, '-r', config$ffm.params$ffm.r, '-t', config$ffm.params$ffm.t.final, '-s', config$ffm.params$ffm.s, ffm.train.fn, ffm.model.fn))
  }

  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    system(paste(ffm.pred.exec, ffm.valid.fn, ffm.model.fn, ffm.preds.fn))
    preds = fread(ffm.preds.fn)$V1

    score = config$eval.score.core.multi(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')

    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }

  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    system(paste(ffm.pred.exec, ffm.test.fn, ffm.model.fn, ffm.preds.fn))
    preds = fread(ffm.preds.fn)$V1

    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

# TODO: add FM
# TODO: add FTRL

train = function(config) {
  if (config$layer == 0 & !is.null(config$finalize.l0.data)) {
    config$finalize.l0.data(config)
  }

  if (config$model.select == 'best') {
    return (train.best(config))
  } else if (config$model.select == 'blend') {
    return (train.blend(config))
  } else if (config$model.select == 'glm') {
    return (train.glm(config))
  } else if (config$model.select == 'nnls') {
    return (train.nnls(config))
  } else if (config$model.select == 'glmnet') {
    return (train.glmnet(config))
  } else if (config$model.select == 'xgb') {
    return (train.xgb(config))
  } else if (config$model.select == 'ranger') {
    return (train.ranger(config))
  } else if (config$model.select == 'et') {
    return (train.et(config))
  } else if (config$model.select == 'rf') {
    return (train.rf(config))
  } else if (config$model.select == 'knn') {
    return (train.knn(config))
  } else if (config$model.select == 'iso') {
    return (train.iso(config))
  } else if (config$model.select == 'nnet') {
    return (train.nnet(config))
  } else if (config$model.select == 'nb') {
    return (train.nb(config))
  } else if (config$model.select == 'gbm') {
    return (train.gbm(config))
  } else if (config$model.select == 'mxnet') {
    return (train.mxnet(config))
  } else if (config$model.select == 'h2o.gbm') {
    return (train.h2o.gbm(config))
  } else if (config$model.select == 'lgb') {
    return (train.lgb(config))
  } else if (config$model.select == 'rgf') {
    return (train.rgf(config))
  } else if (config$model.select == 'vw') {
    return (train.vw(config))
  } else if (config$model.select == 'ffm') {
    return (train.ffm(config))
  } else if (config$model.select == 'custom') {
    return (config$train.custom(config))
  } else {
    stop('wtf')
  }
}

# Cross validation and stacking
# ==================================================================================================

cross.validate = function(config) {
  for (i in 1:config$nr.folds) {
    cat(date(), 'Working on fold', i, '\n')
    config$in.fold = i
    train(config)
  }

  # Examine the overall CV fit
  load(config$ancillary.filename)
  y = ancillary$train.labels
  for (i in 1:config$nr.folds) {
    load(paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', i, '.RData')) # => preds
    if (i == 1) {
      preds = as.matrix(preds)
      x = matrix(NA, length(ancillary$train.labels), ncol(preds))
    }
    x[ancillary$cv.folds[[i]], ] = preds
  }
  score = config$eval.score.core.multi(x, y)
  cat(date(), 'Final CV score:', score, '\n')
  cat(date(), 'Best final CV score:', ifelse(config$score.to.maximize, max(score), min(score)), 'at slot', ifelse(config$score.to.maximize, which.max(score), which.min(score)), '\n')

  cat(date(), 'Working on final model\n')
  config$in.fold = -1
  train(config)

  return (score)
}

# Submission
# ==================================================================================================

generate.submission = function(config) {
  cat(date(), 'Generating submission\n')
  fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
  cat(date(), 'Loading preds from', fnm, '\n')
  load(fnm) # => preds
  load(config$ancillary.filename)

  if (is.matrix(preds)) {
    cat(date(), 'NOTE: submitting column', config$submt.column, '\n')
    preds = preds[, config$submt.column]
  }

  if (!is.null(config$postprocess)) {
    preds = config$postprocess(config, preds)
  }

  submission = data.frame(ID = ancillary$test.ids, pred = preds)
  names(submission) = c(config$id.field, config$target.field)
  readr::write_csv(submission, paste0('sbmt-', config$submt.id, '.csv'))
  zip(paste0('sbmt-', config$submt.id, '.zip'), paste0('sbmt-', config$submt.id, '.csv'))
  cat(date(), ' Submission files created: sbmt-', config$submt.id, '.csv and .zip\n', sep = '')

  if (config$compute.backend != 'condor') {
    ref.sbmt = readr::read_csv(paste0('sbmt-', config$ref.submt.id, '.csv'))
    names(ref.sbmt)[2] = paste0('ref.', config$target.field)
    cmpr.sbmt = merge(ref.sbmt, submission, by = config$id.field)
    if (config$score.select == 'AUC') {
      plot(rank(cmpr.sbmt[[2]]), rank(cmpr.sbmt[[3]]), pch = '.', main = 'Sanity check', xlab = 'Ref pred rank', ylab = 'New pred rank')
    } else {
      plot(cmpr.sbmt[[2]], cmpr.sbmt[[3]], pch = '.', main = 'Sanity check', xlab = 'Ref pred', ylab = 'New pred')
    }
    abline(0, 1, col = 2)
  }
}

# Set up configuration
# ==================================================================================================

create.config = function(score.select, mode = c('single', 'cv', 'cv.batch'), layer = 0) {
  compname = tail(strsplit(getwd(), '/')[[1]], 1)

  config = list()

  config$mode = mode
  config$layer = layer
  config$proper.validation = F # if true, will hold-out some data from entire training process, and test on it

  config$do.preprocess = F
  config$do.train      = F
  config$do.submit     = F

  config$submt.id = 1
  config$submt.column = 1
  config$ref.submt.id = 0

  config$nr.folds = 5
  config$data.seed = 123

  # For illustration purposes:
  if (config$layer == 0) {
    config$model.tag = 'xgb0'
    config$model.select = 'xgb'
    config$rng.seed = 90210

    config$xgb.params = list(
      #booster           = 'gbtree',
      #booster           = 'gblinear',
      #objective         = 'reg:linear',
      #objective         = 'binary:logistic',
      #objective         = 'rank:pairwise',
      #eval_metric       = 'rmse',
      #eval_metric       = 'score',
      nrounds           = 500,
      #eta               = 0.03,
      #max_depth         = 6,
      #min_child_weight  = 1,
      #gamma             = 0,
      #lambda            = 0,
      #alpha             = 0,
      #num_parallel_tree = 1,
      #subsample         = 1,
      #colsample_bytree  = 0.5
      annoying = T
    )

    config$h2o.gbm.params = list(
      distribution             = 'AUTO',
      stopping_metric          = 'AUTO',
      ntrees                   = 500,
      learn_rate               = 0.03,
      max_depth                = 6,
      min_rows                 = 1,
      sample_rate              = 1,
      col_sample_rate_per_tree = 1,
      stopping_rounds          = 0,
      stopping_tolerance       = 0.001,
      annoying = T
    )

    config$glmnet.params = list(
      lambda = exp((-5):(-15)),
      alpha = 1 # i.e., only L1 regularization
    )

    config$rf.params = list(
      ntree = 1000,
      mtry = 10,
      nodesize = 50, # only for randomForest
      max.depth = 50, # only for h2o
      binomial_double_trees = F
    )

    config$ranger.params = list(
      num.trees = 1000,
      mtry = 100
    )

    config$et.params = list(
      ntree = 1000,
      mtry = 100,
      nodesize = 20
    )

    config$knn.params = list(
      k = 11,
      eps = 0
    )

    config$rgf.params = list(
      algorithm = 'RGF', # { RGF, RGF_Opt, RGF_Sib }
      loss = 'LS', # { LS, Expo, Log }
      max_leaf_forest = 500,
      reg_L2 = 0.1,
      reg_sL2 = 0.1,
      min_pop = 10
    )

    # FIXME: add more stuff, need to go over the docs again
    config$vw.params = list(
      loss_function = 'logistic',
      learning_rate = 1e-2,
      interactions = 4
    )

    # FIXME these defualts are from the Outbrain comp and are not the "simplest common case"
    config$lgb.params = list(
      objective        = 'lambdarank',
      max_position     = 12,
      ndcg_at          = 12,
      #label_gain       = 1:12,
      num_iterations   = 1000,
      learning_rate    = 0.1,
      max_depth        = 8,
      min_data_in_leaf = 12,
      feature_fraction = 0.5,
      #num_leaves       = 63,
      #min_sum_hessian_in_leaf = 12,
      #max_bin          = 255,
      early_stopping_rounds = 30,
      eval_freq = 10,
      annoying = T
    )

    config$ffm.params = list(
      ffm.k = 8,
      ffm.l = 0.0001,
      ffm.r = 0.2,
      ffm.s = 16,
      ffm.t.max   = 100,
      ffm.t.final = 10
      # other things: --no-norm, --no-rand, --on-disk if out of RAM
    )
  }

  # Misc training parameters
  config$debug.small = F
  config$holdout.validation = F
  config$measure.importance = F
  config$xgb.early.stop.round = NULL
  config$xgb.ntreelimits.every = 100
  config$id.field = 'ID'
  config$target.field = 'target'

  config$score.select = score.select
  if (score.select == 'logloss') {
    config$target.type = 'factor'
    config$eval.score = eval.logloss
    config$eval.score.ltd = eval.logloss.ltd
    config$eval.score.core = eval.logloss.core
    config$eval.score.core.multi = eval.logloss.core.multi
    config$score.to.maximize = F
  } else if (score.select == 'MAE') {
    config$target.type = 'numeric'
    config$eval.score = eval.mae
    config$eval.score.ltd = eval.mae.ltd
    config$eval.score.core = eval.mae.core
    config$eval.score.core.multi = eval.mae.core.multi
    config$score.to.maximize = F
  } else if (score.select == 'AUC') {
    config$target.type = 'factor'
    config$eval.score = eval.auc
    config$eval.score.ltd = eval.auc.ltd
    config$eval.score.core = eval.auc.core
    config$eval.score.core.multi = eval.auc.core.multi
    config$score.to.maximize = T
  } else if (score.select == 'MAP@3') {
    config$target.type = 'factor'
    config$eval.score = eval.map3
    config$eval.score.ltd = eval.map3.ltd
    config$eval.score.core = eval.map3.core
    config$eval.score.core.multi = eval.map3.core.multi
    config$score.to.maximize = F
  } else if (score.select == 'MAP@5') {
    config$target.type = 'factor'
    config$eval.score = eval.map5
    config$eval.score.ltd = eval.map5.ltd
    config$eval.score.core = eval.map5.core
    config$eval.score.core.multi = eval.map5.core.multi
    config$score.to.maximize = F
  } else if (score.select == 'manual') {
    # caller has/will set these functions manually
  } else {
    stop('Unsupported score.select')
  }

  #
  # Compute platform stuff
  #

  if (.Platform$OS.type == 'windows') {
    config$nr.threads = 7 # I'm using this interactively...
  } else {
    config$nr.threads = parallel::detectCores(all.tests = F, logical = T) # for computation on this machine
  }
  config$h2o.max.mem = '28G'
  config$compute.backend = 'serial' # {serial, multicore, condor, pbs}
  config$nr.cores = ifelse(config$compute.backend %in% c('condor', 'pbs'), 200, config$nr.threads)

  config$package.dependencies = c('ComputeBackend', 'readr', 'xgboost', 'entropy', 'caret', 'Hmisc', 'ranger', 'nnls', 'glmnet')
  config$source.dependencies  = NULL
  config$cluster.dependencies = NULL
  config$cluster.requirements = 'FreeMemoryMB >= 4500'

  config$data.dir = 'input'
  if (.Platform$OS.type == 'windows') {
    config$project.dir = getwd()
  } else {
    config$project.dir = system('pwd', intern = T)
  }
  config$tmp.dir = paste0(config$project.dir, '/tmp')

  dir.create(file.path(config$tmp.dir), showWarnings = F)

  config$dataset.filename   = paste0(config$tmp.dir, '/pp-data-L', config$layer, '.RData')
  config$ancillary.filename = paste0(config$tmp.dir, '/pp-data-ancillary-L', config$layer, '.RData')
  config$xgb.trainset.filename  = paste0(config$tmp.dir, '/xgb-trainset-L', config$layer, '.data')
  config$xgb.testset.filename   = paste0(config$tmp.dir, '/xgb-testset-L', config$layer, '.data')
  config$xgb.validset.filename  = paste0(config$tmp.dir, '/xgb-validset-L', config$layer, '.data')
  config$lgb.trainset.filename  = paste0(config$tmp.dir, '/lgb-trainset-L', config$layer, '.data')
  config$lgb.testset.filename   = paste0(config$tmp.dir, '/lgb-testset-L', config$layer, '.data')
  config$lgb.validset.filename  = paste0(config$tmp.dir, '/lgb-validset-L', config$layer, '.data')

  if (config$debug.small) {
    cat('NOTE: running in small debugging mode\n')
    config$xgb.params$nrounds = 100
  }

  return (config)
}
