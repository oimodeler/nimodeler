


# nimodeler

[![License: GNU](https://img.shields.io/badge/License-GNU-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
![Lifecycle:
Early Development](https://img.shields.io/badge/lifecycle-EarlyDevelopment-orange.svg)

**nimodeler** is an experimental Python package designed to simulate nulling interferometry data compatible with the NIFITS format.

It is built on top of the following libraries:
- **NIFITS**: https://github.com/rlaugier/nifits  
- **oimodeler**: https://github.com/oimodeler/oimodeler/

## Project Status

This package is currently at a very early stage of development.

## Current Features

At its current stage, **nimodeler** allows users to:

- Load a single NIFITS file
- Plot instrument responses for various channels:
  - Photometric  
  - Additive  
  - Nulling
- Extract data from these channels
- Extract the differential nulling channel, when available
- Compute simulated data using an **oimodeler** model with the same instrument response
- Perform data–model comparisons using a simple chi-squared method

![boo](./images/test_nobackground_v3_channels_responses.png)
