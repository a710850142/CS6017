import math
from scipy.stats import norm
def ab_test(clicks_a, views_a, clicks_b, views_b, alpha=0.05, alternative='one-sided'):
    # 计算汇总样本比例
    p_pooled = (clicks_a + clicks_b) / (views_a + views_b)
    # 计算标准误差
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/views_a + 1/views_b))
    # 计算Z值
    p_a = clicks_a / views_a
    p_b = clicks_b / views_b
    z = (p_b - p_a) / se
    # 根据alternative参数计算p值
    if alternative == 'one-sided':
        p_value = 1 - norm.cdf(z)
    else:
        p_value = 2 * (1 - norm.cdf(abs(z)))
    # 与显著性水平比较得出结论
    conclusion = "Reject null hypothesis" if p_value < alpha else "Fail to reject null hypothesis"
    return z, p_value, conclusion
# 原始数据
clicks_a, views_a = 500, 1000
clicks_b, views_b = 550, 1000
# 在5%和1%的显著性水平下进行检验
for alpha in [0.05, 0.01]:
    z, p_value, conclusion = ab_test(clicks_a, views_a, clicks_b, views_b, alpha)
    print(f"Alpha = {alpha}")
    print(f"Z-value: {z:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Conclusion: {conclusion}\n")
# 调整Logo B的点击量以观察结果变化
for clicks_b in [560, 570]:
    print(f"If Logo B clicks = {clicks_b}:")
    z, p_value, conclusion = ab_test(clicks_a, views_a, clicks_b, views_b)
    print(f"Z-value: {z:.4f}")
    print(f"P-value: {p_value:.4f}\n")



