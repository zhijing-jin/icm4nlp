'''
This is the significance test for meta study in Section 5.2 (about semi-supervised learning)
and Section 6 (about domain adaptation).

- Data source: We curated a list of SSL and DA studies in https://docs.google.com/spreadsheets/d/1dNQiFuFMKE05YcTwcnvEZ5xojm3Q2c6c0uR3k7H7D7c/
- Code: We conducted the T-Test using scipy's default function stats.ttest_ind
'''

causal_ssl = [0.8, 1.5, 1.1, 1.2, 0.1, 0.8, 0.8, -0.2, 0.7, -0.1, 0.38, -0.15, -0.05, -0.31, 1.46, 0.44, -1.61, 3.49, 0.22,
          0.68, 1.1, 0.09, -0.02, -0.04, 0, -0.03, -0.82, -1.71, 0.1, 0.76, -0.52, -0.61, 1.49, -0.38, -0.7, -2.06,
          0.28, -23.16, -1.2, -1.74, 0.36, -8.37, -0.74, -1.18, 0.39, -0.1, 0.4, -1.38, 0.94, 11.77, 11.91, 2.9, 1.6,
          0.93, 0.32, 2]
anticausal_ssl = [4.05, 2.31, 3.65, 3.08, 5, 3.45, 4.79, 4.13, 3.17, 1.22, 1.41, 1.58, 1, 0.4, 0.2, 0.5, 0.7, 1.2, 1.2, 1,
              0.4, 0.3, 0, -0.3, 0.1, 0.2, 0.7, 1.88, 0.33, 0.15, -0.2, 1.7, 3.6, 1.37, 0.01, -5.73, 1.21, 2.91, 1.44,
              0.7, 0.6, 1.89, 2.1, 0.21, 1.21, 4.87, 5.24, 5.42, 6.87, 2, ]
causal_da = [1.17, 3.7, 1, 4.9, 1.77, 1.95, 0.96, 1.16, 0.76, 2.3, 0.44, 2.18, 1.92, 0.55, 1.8, 4.2, 10.33, 3.06, 11.58, 23.04, 15.56, 19.76, ]
anticausal_da = [0.4, -0.6, 0.5, 0.8, 3.7, 2.56, 0.94, -0.36, 5.2, ]


def if_significantly_different(result1: list, result2: list, P_VALUE_THRES=0.05):
    from scipy import stats
    import numpy as np

    score, p_value = stats.ttest_ind(result1, np.array(result2), equal_var=False)
    if_sign = p_value <= P_VALUE_THRES
    print('p_value:', p_value)
    return if_sign


if_diff = if_significantly_different(causal_ssl, anticausal_ssl)
print('[Info] Statistical significance of semi-supervised learning (SSL) causal vs. anticausal:', if_diff)

if_diff = if_significantly_different(causal_da, anticausal_da)
print('[Info] Statistical significance of domain adaptation (DA) causal vs. anticausal:', if_diff)


