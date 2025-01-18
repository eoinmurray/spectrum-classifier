# Feature Extraction

> ![](./images/example-spectrum.png) _Figure: Spectrum, we can see clearly
> many sharp peaks coresponding to states in the QD._

Our goal is to define an extraction mapping E from the matrix I to a matrix P
which reduces the dimensionality but retains as much information as possible.

```tex
E: \bold{I}^{m \times n} \to \bold{P}^{i \times j}, \quad \text{where } i < m \text{ and } j < n
```

where m = 23 and n = 2046.

From the spectrum we can extract each peak's center point, amplitude, fwhm
and the skew of the spectrum, whether most of the peaks are on the left or right
of the dominant peak.

From each peak we are able to extract the center point, fwhm, and amplitude.