# Bibliography

The models implemented in LunarOps follow these standards and publications.
Model documentation and docstrings cite them by author and year.

## Conventions and standards

- **Petit & Luzum (2010).** G. Petit, B. Luzum (eds.), *IERS Conventions
  (2010)*, IERS Technical Note No. 36, Verlag des Bundesamts für Kartographie
  und Geodäsie, Frankfurt am Main, 2010. Time scales and their transformations
  (Ch. 10), terrestrial and celestial reference systems (Ch. 4, 5), Earth
  orientation and its tidal and libration variations (Ch. 5, 8), station
  displacements (Ch. 7), tropospheric delay (Ch. 9), relativistic models and
  the gravitational delay (Ch. 11).
- **ILRS CRD v2.** International Laser Ranging Service, *Consolidated Laser
  Ranging Data Format (CRD)*, version 2.
- **IERS EOP products.** IERS Earth Orientation Centre, *EOP 20 C04* (and
  14 C04) series; IERS Rapid Service/Prediction Centre, *finals2000A*
  (Bulletin A).

## Ephemerides and lunar dynamics

- **Folkner et al. (2014).** W. M. Folkner, J. G. Williams, D. H. Boggs,
  R. S. Park, P. Kuchynka, *The Planetary and Lunar Ephemerides DE430 and
  DE431*, IPN Progress Report 42-196, 2014. Lunar orientation and the mantle–core
  rotation model, inertia and torques, Earth tides acting on the Moon.
- **Park et al. (2021).** R. S. Park, W. M. Folkner, J. G. Williams,
  D. H. Boggs, *The JPL Planetary and Lunar Ephemerides DE440 and DE441*,
  The Astronomical Journal 161:105, 2021, doi:10.3847/1538-3881/abd414.
  DE440 force and rotation model, geodetic precession, the Kuiper-belt ring,
  the SB441-N373 small-body set.
- **Pavlov et al. (2016).** D. A. Pavlov, J. G. Williams, V. V. Suvorkin,
  *Determining parameters of Moon's orbital and rotational motion from LLR
  observations using GRAIL and IERS-recommended models*, Celestial Mechanics
  and Dynamical Astronomy 126, 61–88, 2016, doi:10.1007/s10569-016-9712-1.
  LLR observation model, lunar solid tide of the reflectors, delayed lunar
  deformation.
- **Fienga et al. (2021).** A. Fienga et al., *INPOP21a planetary
  ephemerides*, Notes Scientifiques et Techniques de l'Institut de mécanique
  céleste 110, 2021. INPOP21a kernels, station range biases and the
  longitude-libration correction.

## Observation model

- **Mendes & Pavlis (2004).** V. B. Mendes, E. C. Pavlis, *High-accuracy
  zenith delay prediction at optical wavelengths*, Geophysical Research
  Letters 31, L14602, 2004, doi:10.1029/2004GL020308.
- **Mendes et al. (2002).** V. B. Mendes, G. Prates, E. C. Pavlis,
  D. E. Pavlis, R. B. Langley, *Improved mapping functions for atmospheric
  refraction correction in SLR*, Geophysical Research Letters 29(10), 1414,
  2002, doi:10.1029/2001GL014394.

- **Desai (2002).** S. D. Desai, *Observing the pole tide with satellite
  altimetry*, Journal of Geophysical Research 107(C11), 3186, 2002,
  doi:10.1029/2001JC001224. Ocean pole-tide loading coefficients.

## Estimation

- **Yang et al. (2002).** Y. Yang, L. Song, T. Xu, *Robust estimator for
  correlated observations based on bifactor equivalent weights*, Journal of
  Geodesy 76, 353–358, 2002, doi:10.1007/s00190-002-0256-2. The IGG-III weight
  function.
- **Koch (1999).** K.-R. Koch, *Parameter Estimation and Hypothesis Testing in
  Linear Models*, 2nd ed., Springer, 1999. Weighted least squares,
  redundancy numbers and variance-component estimation.

## Software libraries used at run time

- **ERFA.** *Essential Routines for Fundamental Astronomy*, used through
  pyERFA for time scales, precession–nutation and Earth rotation.
- **NAIF SPICE.** NAIF, *DAF Required Reading*, *SPK Required Reading* and
  *PCK Required Reading*, SPICE Toolkit N0067, JPL: the file format and the
  segment types 2, 3 and 20 read by the `chebyshev` ephemeris.

## Numerical methods

- **Battin (1999).** R. H. Battin, *An Introduction to the Mathematics and
  Methods of Astrodynamics*, revised edition, AIAA, 1999. Cancellation-free
  form of the third-body acceleration difference (Sec. 8.4).
- **Dekker (1971).** T. J. Dekker, "A floating-point technique for extending
  the available precision", *Numerische Mathematik* 18, 224-242, 1971.
  Error-free transformations used by the double-double arithmetic.
- **Joldes, Muller & Popescu (2017).** M. Joldes, J.-M. Muller, V. Popescu,
  "Tight and rigorous error bounds for basic building blocks of double-word
  arithmetic", *ACM Transactions on Mathematical Software* 44(2), 2017.
