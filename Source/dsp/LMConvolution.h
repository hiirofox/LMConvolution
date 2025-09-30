#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include "fft.h"

class ConvolutionFIR
{
private:
	std::vector<float> fir;
	std::vector<float> buffer;
	int firlen;
public:
	void SetConvolutionData(float* firdata, int numSamples)
	{
		firlen = numSamples;
		fir.resize(firlen);
		buffer.resize(firlen, 0);
		for (int i = 0; i < firlen; i++)
		{
			fir[i] = firdata[i];
		}
	}
	inline float ProcessSample(float x)
	{
		for (int i = firlen - 1; i > 0; i--)
		{
			buffer[i] = buffer[i - 1];
		}
		buffer[0] = x;
		float y = 0;
		for (int i = 0; i < firlen; i++)
		{
			y += buffer[i] * fir[i];
		}
		return y;
	}
	inline int GetLatencySamples()
	{
		return 0;
	}
};

class ConvolutionFFT
{
private:
	int firlen;//卷积核长度，用于计算预计延迟
	int fftSize;
	std::vector<float> firre, firim;
	std::vector<float> buffer;
	std::vector<float> outre, outim;
	int pos = 0, posHop = 0;

	int nextPowerOfTwo(int n)
	{
		int power = 1;
		while (power < n)
			power *= 2;
		return power;
	}
public:
	void SetConvolutionData(float* fir, int numSamples)//numSamples must be power of 2
	{
		numSamples = nextPowerOfTwo(numSamples);
		firlen = numSamples * 1;
		fftSize = numSamples * 2;
		firre.resize(fftSize, 0);
		firim.resize(fftSize, 0);
		buffer.resize(fftSize, 0);
		outre.resize(fftSize, 0);
		outim.resize(fftSize, 0);
		pos = 0;
		posHop = 0;
		for (int i = 0; i < numSamples; i++)
		{
			firre[i] = fir[i];
		}
		for (int i = numSamples; i < firlen; ++i)
		{
			firre[i] = 0;
		}
		fft_f32(firre, firim, fftSize, 1);
	}
	inline float ProcessSample(float x)
	{
		buffer[pos] = x;
		int lastposhop = posHop;
		pos++, posHop++;
		if (pos >= fftSize)pos = 0;
		if (posHop >= firlen)
		{
			posHop = 0;

			if (pos == 0)
			{
				for (int i = 0; i < fftSize; i++)
				{
					outre[i] = buffer[i];
					outim[i] = 0;
				}
			}
			else
			{
				for (int i = 0; i < fftSize - pos; i++)
				{
					outre[i] = buffer[i + pos];
					outim[i] = 0;
				}
				for (int i = fftSize - pos; i < fftSize; i++)
				{
					outre[i] = buffer[i + pos - fftSize];
					outim[i] = 0;
				}
			}

			fft_f32(outre, outim, fftSize, 1);
			for (int i = 0; i < fftSize; i++)
			{
				float re = outre[i] * firre[i] - outim[i] * firim[i];
				float im = outre[i] * firim[i] + outim[i] * firre[i];
				outre[i] = re;
				outim[i] = im;
			}
			fft_f32(outre, outim, fftSize, -1);
			for (int i = 0; i < fftSize; i++)
			{
				outre[i] /= fftSize;
			}
		}
		return outre[posHop + firlen];
	}

	inline int GetLatencySamples()
	{
		return firlen;
	}
};

class TestConvolution
{
public:
	constexpr static int TestLen = 2048;
private:
	ConvolutionFFT cfft;
	float testdatre[TestLen];
	float testdatim[TestLen];
public:
	TestConvolution()
	{
		auto randf = []() { return (float)(rand() % 10000) / 10000.0 * (rand() % 2 ? 1 : -1); };
		for (int i = 0; i < TestLen; i++)
		{
			testdatre[i] = randf();
			testdatim[i] = 0;
		}
		//testdatre[0] = 1.0;
		//testdatre[TestLen - 1] = 1.0;
		cfft.SetConvolutionData(testdatre, TestLen);
	}
	void ProcessBlock(const float* in, float* out, int numSamples)
	{
		for (int i = 0; i < numSamples; i++)
		{
			out[i] = cfft.ProcessSample(in[i]);
		}
	}
};