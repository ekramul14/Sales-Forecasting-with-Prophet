# Holdout evaluation

Stores evaluated: 50 | Holdout: last 42 days (2015-06-20 to 2015-07-31), open days with sales > 0 | Models trained only on data before the holdout.

| Model | Mean MAPE (%) | Median MAPE (%) | Mean RMSPE (%) |
|---|---|---|---|
| naive_last_week | 24.20 | 23.72 | 32.76 |
| weekday_avg_8w | 21.20 | 21.03 | 26.33 |
| prophet_holidays | 19.02 | 18.74 | 23.06 |
| prophet_promo | 11.33 | 10.44 | 14.30 |

Best Prophet model (prophet_promo) vs best baseline (weekday_avg_8w): mean MAPE 11.33% vs 21.20%, a 46.6% relative reduction in error; Prophet had the lower MAPE in 98% of stores.
