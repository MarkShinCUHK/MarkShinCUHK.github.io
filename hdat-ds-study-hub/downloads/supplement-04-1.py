from math import erf, exp, isclose, pi, sqrt

def expected_improvement(mu, sigma, best, xi=0.0):
    """점수 최대화 문제의 단일 후보 EI. Gaussian surrogate 가정."""
    if sigma < 0:
        raise ValueError("표준편차는 음수일 수 없습니다")
    improvement = mu - best - xi
    if sigma == 0:
        return max(improvement, 0.0)
    z = improvement / sigma
    cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    density = exp(-0.5 * z*z) / sqrt(2.0*pi)
    return max(improvement * cdf + sigma * density, 0.0)

assert isclose(expected_improvement(0.8, 0.0, 0.7), 0.1)
assert expected_improvement(0.6, 0.0, 0.7) == 0.0
assert isclose(expected_improvement(0.7, 0.1, 0.7), 0.1/sqrt(2*pi))
# 같은 평균이라면 아직 불확실한 후보에도 개선 가능성이 있다.
assert expected_improvement(0.7, 0.1, 0.7) > expected_improvement(0.7, 0, 0.7)
print("EI 경계값·손계산 검사 통과")
