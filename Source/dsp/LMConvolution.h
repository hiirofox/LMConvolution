#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include "fft.h"

class ConvolutionFIR
{
private:
	std::vector<float> fir;
	std::vector<float> buffer;
	int firlen = 0;
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
	int firlen = 0;//卷积核长度，用于计算预计延迟
	int fftSize = 0;
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

class DelayLine
{
private:
	std::vector<float> buffer;
	int bufsize = 0;
	int writepos = 0;
public:
	void SetDelayTime(float numSamples)
	{
		bufsize = numSamples + 1;
		buffer.resize(bufsize, 0);
		writepos = 0;
	};
	inline float ProcessSample(float x)
	{
		buffer[writepos] = x;
		writepos = (writepos + 1) % bufsize;
		return buffer[writepos];
	}
};

//https://publications.rwth-aachen.de/record/466561/files/466561.pdf?subformat=pdfa
class LMConvolution1
{
private:
	constexpr static int firlen = 32;//must be power of 2
	constexpr static int MaxStages = 256;
	ConvolutionFIR cfir;
	std::vector<ConvolutionFFT> cffts;
	std::vector<DelayLine> delays;
	int numStages = 0;
public:
	LMConvolution1()
	{
		cffts.resize(MaxStages);//32已经很长了
		delays.resize(MaxStages);
	}
	void SetConvolutionData(float* fir, int numSamples)
	{
		numStages = 0;
		cfir.SetConvolutionData(fir, std::min(numSamples, firlen));
		if (numSamples <= firlen)return;
		numSamples -= firlen;
		for (int i = 0, pos = firlen; i < MaxStages; ++i)
		{
			int len = firlen << i;
			//if (len > 8192)len = 8192;
			cffts[i].SetConvolutionData(&fir[pos], std::min(numSamples, len));
			delays[i].SetDelayTime(pos - cffts[i].GetLatencySamples());
			if (numSamples <= len)
			{
				numStages = i + 1;
				return;
			}
			numSamples -= len;
			pos += len;
		}
	}
	inline float ProcessSample(float x)
	{
		float y = cfir.ProcessSample(x);
		for (int i = 0; i < numStages; i++)
		{
			y += cffts[i].ProcessSample(delays[i].ProcessSample(x));
		}
		return y;
	}
};

class TestConvolution
{
public:
	constexpr static int TestLen = 65536;
private:
	LMConvolution1 convolution;
	float testdatre[TestLen];
	float testdatim[TestLen];
public:
	TestConvolution()
	{
		auto randf = []() { return (float)(rand() % 10000) / 10000.0 * (rand() % 2 ? 1 : -1); };
		for (int i = 0; i < TestLen; i++)
		{
			float x = (float)(i + 100) / (TestLen + 100);
			x = x * x;
			testdatre[i] = sinf(x * 2.0 * M_PI * 10000.0);
			testdatim[i] = 0;
		}
		//testdatre[0] = 1.0;
		//testdatre[TestLen / 2] = 1.0;
		convolution.SetConvolutionData(testdatre, TestLen);
	}
	float fbv = 0;
	void ProcessBlock(const float* in, float* out, int numSamples)
	{
		for (int i = 0; i < numSamples; i++)
		{
			out[i] = fbv = convolution.ProcessSample(in[i] - fbv * 0.0025);
		}
	}
};