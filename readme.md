# Project Title
Creation of Symbolic kern dataset for Hindustani Classical Music:

## Problem Statement
Given the Hindustani Classical Music Composition in Bhatkhande notation in Scanned document, The goal is to develop a methodology to create a symbolic kern dataset for the composition.

## Tools required
python environment and libraries installed in your system

## Pre-requisites
1. Understanding of Music Notations:
Understanding of music theory is essential, especially regarding scales, intervals, rhythms, and notation systems like Bhatkhande notation and kern notation. This includes understanding the symbols, conventions, and rules of each system.

2. Image Processing Techniques and ML algorithms.
Understanding of Lanczos interpolation, Gaussian blur, Sharpening filter, Bounding box for segmentation, and ML algorithms like k-means clustering, HAC (Hierarchical Agglomerative Clustering). Basics of training CNN (Convolution Neural Network) model and evaluation of model for recognition of the Devanagari alphabet.

## Description
We have a scanned PDF of the music composition in Bhatkhande notation. we need to write the composition in .kern (Humdrum) file. 

### Step 1:
We start with the segmentation of each alphabet as individual image using contours and bounding box. For recognition of these alphabets, we need to train CNN Model. So, we prepare pdf of compositions which covers different alphabets, and then apply segmentation.

### Step 2:
Apply K-means clustering or HAC on the segmented images and name the appropriate names to the clusters of alphabets. Apply Image augmentation and create different images and prepare a dataset for further training. Convert the prepared dataset of images to CSV file with class label.

### Step 3:
Train and evaluate CNN model using dataset created. 

### Step 4:
Given PDF of the music composition in Bhatkhande notation, apply segmentation. Recognize swar and text sequences from the pdf, and identify alphabet using model. 