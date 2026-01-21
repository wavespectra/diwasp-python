# Estimation Methods

DIWASP provides five different methods for estimating directional wave spectra. Each method has different characteristics in terms of accuracy, speed, and suitability for different data types.

## Method Overview

| Method | Speed | Accuracy | Noise Tolerance | Best For |
|--------|-------|----------|-----------------|----------|
| DFTM | Very Fast | Low | Poor | Quick overview |
| EMLM | Fast | Medium | Poor | Narrow spectra |
| IMLM | Medium | Good | Medium | General use |
| EMEP | Variable | Very Good | Good | Noisy data |
| BDM | Slow | Excellent | Excellent | Best accuracy |

## DFTM: Direct Fourier Transform Method

**Reference:** Barber (1961)

The DFTM is the simplest and fastest method. It directly integrates the cross-spectra with transfer functions.

**Algorithm:**

```
S(f, theta) ~ sum_n sum_m [H_n * H_m* * C_nm * exp(i * kx_nm)]
```

**Characteristics:**

- Very fast, non-iterative
- Good for initial overview of spectral shape
- Poor directional resolution
- Can produce negative energy (unphysical)
- Poor tolerance of errors in data

**When to use:**

- Quick sanity check of data
- Initial exploration before using better methods
- When computation speed is critical

## EMLM: Extended Maximum Likelihood Method

**Reference:** Isobe et al. (1984)

The EMLM inverts the cross-spectral density matrix to improve directional resolution.

**Algorithm:**

```
E(theta) = 1 / sum_nm [H_n * H_m* * C_inv_nm * exp(i * kx_nm)]
```

**Characteristics:**

- Fast method
- Good accuracy with narrow unidirectional spectra
- Can provide excellent accuracy per computation time in ideal cases
- Poor tolerance of errors can lead to negative energy or failure
- Sensitive to noise due to matrix inversion

**When to use:**

- Clean data with narrow directional spreading
- When moderate speed is needed with better resolution than DFTM

## IMLM: Iterated Maximum Likelihood Method

**Reference:** Pawka (1983)

The IMLM iteratively refines the EMLM estimate.

**Algorithm:**

1. Start with initial estimate (e.g., from EMLM)
2. Iteratively update using weighted likelihood
3. Apply relaxation to ensure convergence

**Parameters:**

- `max_iter`: Number of improvement iterations (default 100)
- `gamma`: Step size relaxation (default 0.1)
- `alpha`: Convergence factor (default 0.1)

**Characteristics:**

- Computation time directly depends on iteration count
- Reduces anomalies like negative energy from EMLM
- Can overestimate peaks by overcorrecting
- Quality depends on initial EMLM solution

**When to use:**

- General purpose estimation
- When EMLM gives reasonable but imperfect results
- Good balance of speed and accuracy

## EMEP: Extended Maximum Entropy Principle

**Reference:** Hashimoto et al. (1993)

The EMEP uses model selection based on the Akaike Information Criterion (AIC).

**Algorithm:**

1. Separate co- and quadrature components from cross-spectra
2. Use cosine/sine basis expansion with unknown coefficients
3. Iteratively fit models of increasing order
4. Select best model using AIC

**Characteristics:**

- Good all-round method that accounts for errors
- Computation time is highly variable
- Can be as fast as IMLM with superior results
- Low energies at spectral tails can slow computation
- Automatic model order selection provides robustness

**When to use:**

- **Default recommended method**
- Data with significant noise or errors
- When robustness is more important than speed
- Multi-modal spectra

## BDM: Bayesian Direct Method

**Reference:** Hashimoto and Kobune (1987)

The BDM uses Bayesian inference with Tikhonov regularization.

**Algorithm:**

1. Log-transform spectral density: `x = log(S)`
2. Solve inverse problem with Tikhonov regularization
3. Use QR decomposition for numerical stability
4. Select regularization via Akaike Bayesian IC (ABIC)

**Characteristics:**

- Overall best accuracy
- Very computationally intensive
- Natural smoothing via Laplacian regularization
- Robust to noise
- Can have problems with three-quantity measurements (PUV, heave-roll-pitch)

**When to use:**

- When accuracy is paramount and time permits
- Noisy or complex data
- Research applications requiring best estimates

## Choosing a Method

### Decision Flowchart

```
Is this a quick sanity check?
├── Yes → DFTM
└── No → Is the data clean with narrow spreading?
         ├── Yes → EMLM or IMLM
         └── No → Is computation time critical?
                  ├── Yes → EMEP
                  └── No → BDM
```

### Tips for Each Method

**All methods:**
- Reduce frequency resolution to increase speed

**EMEP/BDM:**
- Reduce directional resolution (`dres`) to increase speed
- Optimal iteration count before relaxation varies by dataset

**Testing:**
Use synthetic data from `makespec` to test methods for your specific instrument configuration before processing real data.

## Resolution Parameters

### EP.nfft - FFT Length

Controls frequency resolution. Higher values give finer frequency bins but require more data and computation.

```
frequency_resolution = fs / nfft
```

### EP.dres - Directional Resolution

Number of directional bins covering 360 degrees. Default is 180 (2-degree bins).

Reducing this value dramatically improves speed for EMEP and BDM.

### EP.iter - Iterations

For **IMLM**: Number of improvement corrections at each frequency. Directly affects computation time.

For **EMEP/BDM**: Limit before computation "relaxes" the iterative calculation. Reducing doesn't necessarily improve speed if the algorithm converges before reaching the limit.

### EP.smooth - Smoothing

Applies 2D smoothing to remove unphysical spikes. Recommended to keep enabled (default).

## Troubleshooting

### Garbage Output

If you get nonsensical results:

1. Try DFTM first - it rarely fails completely
2. If DFTM gives garbage, check your inputs
3. Verify sensor positions and types are correct
4. Check data quality and synchronization

### Slow Computation (EMEP/BDM)

Low spectral energies at high/low frequencies can cause slow convergence:

1. Reduce directional resolution
2. Limit frequency range to energetic portion
3. Reduce iteration limit (may reduce quality)

### Negative Energy

If you see negative spectral values:

1. Switch from DFTM/EMLM to EMEP or BDM
2. These methods naturally prevent negative values
3. Check data for errors or contamination

### Matrix Warnings

Warnings about matrix conditioning during EMLM or IMLM:

1. Usually handled internally
2. Switch to EMEP or BDM for more robust handling
3. May indicate too few sensors or poor geometry
