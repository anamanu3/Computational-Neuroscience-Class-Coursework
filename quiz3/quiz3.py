import mpmath as mp

# Params: r|s1 ~ N(mu1, sigma1^2), r|s2 ~ N(mu2, sigma2^2)
mu1, sigma1 = 5.0, 0.5
mu2, sigma2 = 7.0, 1.0

# Asymmetric-cost likelihood ratio: eta = C21 / C12
eta = 2.0  # twice as bad to mistake s1 as s2

def gaussian_pdf(x, mu, sigma):
    return (1.0/(sigma*mp.sqrt(2*mp.pi))) * mp.e**(- (x-mu)**2 / (2*sigma**2))

def equation(r):
    return gaussian_pdf(r, mu2, sigma2) / gaussian_pdf(r, mu1, sigma1) - eta

# Solve for r* (pick a seed between the means)
r_star = mp.findroot(equation, 5.9)

# Convert to float for printing
print(f"Decision threshold r*: {float(r_star):.6f}")