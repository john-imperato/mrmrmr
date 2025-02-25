functions {
  // compute the (T, 3) matrix of forward probabilities for one individual
  matrix forward_prob(int i, // individual index
                      matrix X_surv, vector beta_phi, vector eps_phi,
                      vector logit_detect, vector lambda, vector gam_init,
                      array[] int introduced, array[] int t_intro,
                      array[] int removed, array[] int t_remove,
                      array[] int prim_idx, array[] int any_surveys,
                      array[] int J, array[,] int j_idx, array[,,] int Y,
                      int Jtot, int T) {
    // define temporary scope vars for function
    array[3] real acc;
    array[T] vector[3] gam;
    vector[T] phi;
    matrix[T, 3] forward;
    array[3, T, 3] real ps;
    array[3, Jtot, 3] real po;
    real p;
    // s = 1 :: not recruited
    // s = 2 :: alive
    // s = 3 :: dead
    
    // define probs of state S(t+1) | S(t)
    // first index: S(t)
    // second index: individual
    // third index: t
    // fourth index: S(t + 1)
    for (t in 1 : T) {
      phi[t] = inv_logit(X_surv[i,  : ] * beta_phi + eps_phi[t]);
      ps[1, t, 3] = 0; // can't die before being alive
      ps[2, t, 1] = 0; // can't unenter population
      ps[3, t, 1] = 0;
      ps[2, t, 2] = phi[t]; // survive
      ps[2, t, 3] = 1 - phi[t]; // death
      ps[3, t, 2] = 0; // cannot un-die
      ps[3, t, 3] = 1; // dead stay dead
    }
    
    if (introduced[i]) {
      // individual has been introduced
      // zero probability of recruiting through t_intro - 1
      for (t in 1 : (t_intro[i] - 1)) {
        ps[1, t, 1] = 1;
        ps[1, t, 2] = 0;
        ps[1, t, 3] = 0;
      }
      
      // timestep of introduction has Pr(recruiting) = 1
      ps[1, t_intro[i], 1] = 0;
      ps[1, t_intro[i], 2] = 1;
      ps[1, t_intro[i], 3] = 0;
      
      // to avoid NA values, fill in remaining recruitment probs (though they
      // are irrelevant for the likelihood)
      for (t in (t_intro[i] + 1) : T) {
        ps[1, t, 1] = 1;
        ps[1, t, 2] = 0;
        ps[1, t, 3] = 0;
      }
    } else {
      for (t in 1 : T) {
        ps[1, t, 1] = 1 - lambda[t];
        ps[1, t, 2] = lambda[t];
        ps[1, t, 3] = 0;
      }
    }
    
    if (removed[i]) {
      if (t_remove[i] < T) {
        ps[2, t_remove[i] + 1, 2] = 0;
        ps[2, t_remove[i] + 1, 3] = 1;
      }
    }
    
    // observation probabilities
    for (j in 1 : Jtot) {
      p = inv_logit(logit_detect[j]);
      po[1, j, 1] = 1;
      po[1, j, 2] = 0;
      
      if (prim_idx[j] == t_intro[i]) {
        // introductions always happen after surveys, so if an individual is
        // released on primary period t, it has a zero probability of
        // detection
        po[2, j, 1] = 1;
        po[2, j, 2] = 0;
      } else {
        po[2, j, 1] = 1 - p;
        po[2, j, 2] = p;
      }
      
      po[3, j, 1] = 1;
      po[3, j, 2] = 0;
    }
    
    for (t in 1 : T) {
      for (k in 1 : 3) {
        for (kk in 1 : 3) {
          if (t == 1) {
            acc[kk] = gam_init[kk];
          } else {
            acc[kk] = gam[t - 1, kk];
          }
          acc[kk] *= ps[kk, t, k];
          if (any_surveys[t]) {
            for (j in 1 : J[t]) {
              // only increment the probability with the likelihood if surveys happened
              acc[kk] *= po[k, j_idx[t, j], Y[i, t, j]];
            }
          }
        }
        gam[t, k] = sum(acc);
      }
      forward[t,  : ] = gam[t,  : ]';
    }
    return forward;
  }
  
  real partial_sum_lpmf(array[] int slice_individuals, int start, int end,
                        matrix X_surv, vector beta_phi, vector eps_phi,
                        vector logit_detect, vector lambda, vector gam_init,
                        array[] int introduced, array[] int t_intro,
                        array[] int removed, array[] int t_remove,
                        array[] int prim_idx, array[] int any_surveys,
                        array[] int J, array[,] int j_idx, array[,,] int Y,
                        int Jtot, int T) {
    real loglik = 0;
    matrix[T, 3] forward;
    
    for (i in start : end) {
      forward = forward_prob(i, X_surv, beta_phi, eps_phi, logit_detect,
                             lambda, gam_init, introduced, t_intro, removed,
                             t_remove, prim_idx, any_surveys, J, j_idx, Y,
                             Jtot, T);
      
      loglik += log(sum(forward[T,  : ]));
    }
    
    return loglik;
  }
}
//============================================================================
//============================================================================
data {
  int<lower=1> M; // augmented sample size
  int<lower=1> T; // number of primary periods
  int<lower=1> maxJ; // max number of secondary periods
  array[T] int<lower=0, upper=maxJ> J; // number of secondary periods for each primary period
  int<lower=1> Jtot; // total number of surveys
  array[Jtot] int<lower=1, upper=T> prim_idx; // mapping from survey to primary period
  
  // observations: 0 = NA, 1 = not detected, 2 = detected
  array[M, T, maxJ] int<lower=0, upper=2> Y;
  array[M] int<lower=0, upper=1> introduced; // indicator for whether introduced
  array[M] int<lower=0, upper=T> t_intro; // when individuals introduced
  array[M] int<lower=0, upper=1> removed;
  array[M] int<lower=0, upper=T> t_remove;
  
  // index order of surveys (0: NA)
  array[T, maxJ] int<lower=0, upper=Jtot> j_idx;
  array[T] int<lower=0, upper=1> any_surveys;
  
  // fixed effects design matrices
  int<lower=1> m_detect;
  matrix[Jtot, m_detect] X_detect;
  int<lower=1> m_surv;
  matrix[M, m_surv] X_surv;
  
  int<lower=0, upper=1> any_recruitment;
  int grainsize;
}
//============================================================================
//============================================================================
transformed data {
  array[M] int Mseq;
  int Tm1 = T - 1; // number of primary periods - 1
  vector[3] gam_init = [1, 0, 0]';
  
  for (i in 1 : M) {
    Mseq[i] = i;
  }
}
//============================================================================
//============================================================================
parameters {
  // recruitment parameters
  real alpha_lambda;
  real<lower=0> sigma_lambda;
  vector[T] eps_lambda;
  
  // survival parameters
  vector[m_surv] beta_phi;
  real<lower=0> sigma_phi;
  vector[T] eps_phi;  // random effect on survival per primary period
  
  // detection parameters
  vector[m_detect] beta_detect;
    // <<--2/10/25: added for random effects on detection probability
  real<lower=0> sigma_detect;
  vector[Jtot] eps_detect;
}
//============================================================================
//============================================================================
transformed parameters {
  vector[Jtot] logit_detect;
  vector<lower=0, upper=1>[T] lambda;
  
  // probability of entering population (recruitment)
  lambda = any_recruitment
           * inv_logit(alpha_lambda + eps_lambda * sigma_lambda);
  
  // probability of detection including the session-specific random effect:
  // For each survey j, add the associated random effect
  for (j in 1:Jtot) {
    logit_detect[j] = X_detect[j] * beta_detect + eps_detect[j];
  }
}
//============================================================================
//============================================================================
model {
  // priors for recruitment
  alpha_lambda ~ std_normal();
  sigma_lambda ~ std_normal();
  eps_lambda ~ std_normal();
  
  // priors for detection fixed effects
  beta_detect ~ std_normal();
  // <<-- 2/10/25: priors for detection random effects:
  sigma_detect ~ std_normal();
  eps_detect ~ normal(0, sigma_detect);
  
  // priors for survival
  beta_phi ~ std_normal();
  sigma_phi ~ std_normal();
  eps_phi ~ normal(0, sigma_phi);
  
  target += reduce_sum(partial_sum_lpmf, Mseq, grainsize, X_surv, beta_phi,
                       eps_phi, logit_detect, lambda, gam_init, introduced,
                       t_intro, removed, t_remove, prim_idx, any_surveys, J,
                       j_idx, Y, Jtot, T);
}

// ============================================================================
// ============================================================================

generated quantities {
  // Latent states for each individual and primary period:
  array[M, T] int<lower=1, upper=3> s;
  // Superpopulation size:
  int<lower=0> Nsuper;
  // Abundance and recruitment for periods 2 through T:
  array[Tm1] int<lower=0> N;  // where Tm1 = T - 1 (set in transformed data block)
  array[Tm1] int<lower=0> B;
  
  // Define overall survival for transitions between the "real" periods.
  // Period 1 is a dummy, so the real survey periods are 2,...,T,
  // so there are (T - 2) transitions (from period 2 to 3, ..., T-1 to T).
  vector[T-2] overall_phi;  // average survival for each transition
  
  // Detection probability for each survey
  vector[Jtot] p;
  // Log-likelihood for each individual:
  vector[M] log_lik;
  
  // ---------------------------------------------------------------
  // Compute detection probabilities for each survey.
  for (j in 1:Jtot) {
    p[j] = inv_logit(X_detect[j] * beta_detect + eps_detect[j]);
  }
  
  // ---------------------------------------------------------------
  // Compute overall survival probabilities for the transitions between the real periods for only real individuals.
  // That is, for t = 2,...,T-1, compute survival from period t to t+1.
  // overall_phi[1] corresponds to survival from period 2 to 3, etc.
for (t in 2:(T - 1)) {
  real sum_phi = 0;
  int real_count = 0;
  for (i in 1:M) {
    // w[i] is 1 for real individuals, 0 for ghosts
    if (w[i] == 1) {
      sum_phi += inv_logit(X_surv[i] * beta_phi + eps_phi[t]);
      real_count += 1;
    }
  }
  overall_phi[t] = sum_phi / real_count;
}
  
  // ---------------------------------------------------------------
  // Compute latent states (s) and individual log-likelihoods using the forward algorithm.
  {
    array[3, T, 3] real ps;
    vector[T] phi;
    vector[3] tmp;
    matrix[T, 3] forward_probabilities;
    
    for (i in 1:M) {
      // Compute survival probabilities for each period.
      for (t in 1:T) {
        phi[t] = inv_logit(X_surv[i] * beta_phi + eps_phi[t]);
        // Set up state transition probabilities:
        ps[1, t, 3] = 0;
        ps[2, t, 1] = 0;
        ps[3, t, 1] = 0;
        ps[2, t, 2] = phi[t];
        ps[2, t, 3] = 1 - phi[t];
        ps[3, t, 2] = 0;
        ps[3, t, 3] = 1;
      }
      
      // Adjust recruitment probabilities for introduced individuals.
      if (introduced[i]) {
        for (t in 1:(t_intro[i] - 1)) {
          ps[1, t, 1] = 1;
          ps[1, t, 2] = 0;
          ps[1, t, 3] = 0;
        }
        ps[1, t_intro[i], 1] = 0;
        ps[1, t_intro[i], 2] = 1;
        ps[1, t_intro[i], 3] = 0;
        for (t in (t_intro[i] + 1):T) {
          ps[1, t, 1] = 1;
          ps[1, t, 2] = 0;
          ps[1, t, 3] = 0;
        }
      } else {
        for (t in 1:T) {
          ps[1, t, 1] = 1 - lambda[t];
          ps[1, t, 2] = lambda[t];
          ps[1, t, 3] = 0;
        }
      }
      
      // Adjust for removals.
      if (removed[i]) {
        if (t_remove[i] < T) {
          ps[2, t_remove[i] + 1, 2] = 0;
          ps[2, t_remove[i] + 1, 3] = 1;
        }
      }
      
      // Compute the forward probabilities using the provided function.
      forward_probabilities = forward_prob(i, X_surv, beta_phi, eps_phi,
                                           logit_detect, lambda, gam_init,
                                           introduced, t_intro, removed,
                                           t_remove, prim_idx, any_surveys,
                                           J, j_idx, Y, Jtot, T);
      
      // Backward sampling to obtain latent state trajectory for individual i.
      s[i, T] = categorical_rng(forward_probabilities[T, ]' / sum(forward_probabilities[T, ]));
      for (t_rev in 1:Tm1) {
        int t = T - t_rev;
        int tp1 = t + 1;
        tmp = forward_probabilities[t, ]' .* to_vector(ps[ , tp1, s[i, tp1]]);
        s[i, t] = categorical_rng(tmp / sum(tmp));
      }
      
      // Record the individual log-likelihood.
      log_lik[i] = log(sum(forward_probabilities[T, ]));
    }
  }
  
  // ---------------------------------------------------------------
  // Compute derived population-level quantities (for periods 2 through T).
  {
    array[M, Tm1] int al;
    array[M, Tm1] int d;
    array[M] int alive;
    array[M] int w;
    
    for (i in 1:M) {
      // For each individual, record whether it is alive (state 2) in periods 2 to T.
      for (t in 2:T) {
        al[i, t - 1] = s[i, t] == 2;
      }
      for (t in 1:Tm1) {
        d[i, t] = s[i, t] == al[i, t];
      }
      alive[i] = sum(al[i]);
    }
    
    for (t in 1:Tm1) {
      N[t] = sum(al[ , t]);
      B[t] = sum(d[ , t]);
    }
    
    for (i in 1:M) {
      w[i] = 1 - !alive[i];
    }
    Nsuper = sum(w);
  }
}
