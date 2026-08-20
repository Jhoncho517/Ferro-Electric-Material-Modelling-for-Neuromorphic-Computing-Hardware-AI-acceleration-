# Ferro-Electric Material Modelling for Neuromorphic Computing Hardware AI Acceleration

## Project Overview

As traditional computing struggles to keep up with the energy demands of modern AI, researchers are looking to brain-inspired ("neuromorphic") hardware that stores and processes information more efficiently. Ferroelectric transistors made from hafnium oxide (HfO₂) are strong candidates because they can hold multiple memory states, much like the adjustable connections between brain cells.

### The Challenge

The challenge is that these materials switch in a random, unpredictable way: the material is made of many tiny grains, and each one flips at a slightly different point, giving the device a "memory" of its past that is hard to predict. Without an accurate model of this behavior, engineers cannot reliably design these devices for computing.

### Our Solution

This project builds a simulation that captures that randomness directly. Using a Monte Carlo method, we model 5,000 individual grains and treat each one's switching as a probability that depends on the applied voltage and the grain's own history. We then connect this to a transistor model so the simulation runs all the way from an input voltage pulse to the device's final electrical response.

### Key Results

The results show that the device's memory state can be tuned by changing the strength and length of the voltage pulse—the same kind of gradual, adjustable behavior that lets brain synapses learn. Looking ahead, this framework gives device designers a practical, physically realistic tool for building HfO₂ transistors into neuromorphic chips, supporting future low-power hardware for AI, edge computing, and on-device learning.

---

## Documentation

### View Monte Carlo FeF Document

Open the poster in Google Docs viewer (recommended):

https://docs.google.com/gview?url=https://raw.githubusercontent.com/Jhoncho517/Ferro-Electric-Material-Modelling-for-Neuromorphic-Computing-Hardware-AI-acceleration-/main/Monte_Carlo_FeFET_Poster.pdf

You can also embed it using an iframe (note: GitHub README pages sanitize and often do not render iframes; the direct link above will open the viewer in a new tab):

<iframe src="https://docs.google.com/gview?url=https://raw.githubusercontent.com/Jhoncho517/Ferro-Electric-Material-Modelling-for-Neuromorphic-Computing-Hardware-AI-acceleration-/main/Monte_Carlo_FeFET_Poster.pdf"></iframe>

Or [download the PDF directly](https://github.com/Jhoncho517/Ferro-Electric-Material-Modelling-for-Neuromorphic-Computing-Hardware-AI-acceleration-/raw/main/Monte_Carlo_FeFET_Poster.pdf)
