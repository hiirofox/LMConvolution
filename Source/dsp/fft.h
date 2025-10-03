#pragma once

#include <math.h>
#include <vector>
#include <complex>

void fft_f32(float* are, float* aim, int n, int inv);
void fft_f32(std::vector<float>& are, std::vector<float>& aim, int n, int inv);
void fft_f32(std::vector<std::complex<float>>& a, int n, int inv);