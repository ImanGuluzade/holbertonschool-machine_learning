# Probability Distributions

A collection of Python classes representing various mathematical distributions. Each distribution is implemented from scratch without using external math libraries to demonstrate the underlying probability theory.

## Distributions Covered

* **Poisson**: Discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time.
* **Exponential**: Probability distribution of the time between events in a Poisson point process.
* **Normal**: Also known as the Gaussian distribution; represents the "bell curve."
* **Binomial**: Discrete probability distribution of the number of successes in a sequence of $n$ independent experiments.

## Mathematical Formulas Used

### Normal Distribution PDF
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$



### Binomial PMF
$$P(k) = \binom{n}{k} p^k (1-p)^{n-k}$$



[Image of Binomial distribution PMF formula]


## Requirements
- Language: Python 3.5
- Style: `pycodestyle` (PEP 8)
- No external module imports (e.g., `math`, `numpy`)

## Author
* **Your Name** - [Your GitHub Profile]
