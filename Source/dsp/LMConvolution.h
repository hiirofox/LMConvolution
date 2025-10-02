#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <atomic>
#include <thread>
#include <JuceHeader.h>

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
	int firlen = 0;
	int fftSize = 0;
	int fftOrder = 0;

	std::unique_ptr<juce::dsp::FFT> fft;

	std::vector<float> firre, firim;
	std::vector<float> buffer;
	std::vector<float> outre, outim;
	std::vector<float> fftBuffer; // JUCE FFT需要交错格式的缓冲区

	int pos = 0, posHop = 0;

	int nextPowerOfTwo(int n)
	{
		int power = 1;
		while (power < n)
			power *= 2;
		return power;
	}

	int log2Int(int n)
	{
		int order = 0;
		while (n > 1)
		{
			n >>= 1;
			order++;
		}
		return order;
	}

public:
	void SetConvolutionData(float* fir, int numSamples)
	{
		numSamples = nextPowerOfTwo(numSamples);
		firlen = numSamples * 1;
		fftSize = numSamples * 2;
		fftOrder = log2Int(fftSize);

		// 创建JUCE FFT对象
		fft = std::make_unique<juce::dsp::FFT>(fftOrder);

		firre.resize(fftSize, 0);
		firim.resize(fftSize, 0);
		buffer.resize(fftSize, 0);
		outre.resize(fftSize, 0);
		outim.resize(fftSize, 0);
		fftBuffer.resize(fftSize * 2, 0); // 交错格式: [re0, im0, re1, im1, ...]

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

		// 将实部虚部转换为JUCE的交错格式
		for (int i = 0; i < fftSize; ++i)
		{
			fftBuffer[i * 2] = firre[i];
			fftBuffer[i * 2 + 1] = 0;
		}

		// 执行FFT
		fft->performRealOnlyForwardTransform(fftBuffer.data(), true);

		// 转换回实部虚部格式保存
		for (int i = 0; i < fftSize; ++i)
		{
			firre[i] = fftBuffer[i * 2];
			firim[i] = fftBuffer[i * 2 + 1];
		}
	}

	// 新增:准备FFT计算(用于多线程)
	inline bool NeedFFTCompute()
	{
		return (posHop == 0);
	}

	// 新增:执行FFT计算(可在独立线程中调用)
	inline void ComputeFFT()
	{
		// 准备输入数据
		if (pos == 0)
		{
			for (int i = 0; i < fftSize; i++)
			{
				outre[i] = buffer[i];
			}
		}
		else
		{
			for (int i = 0; i < fftSize - pos; i++)
			{
				outre[i] = buffer[i + pos];
			}
			for (int i = fftSize - pos; i < fftSize; i++)
			{
				outre[i] = buffer[i + pos - fftSize];
			}
		}

		// 转换为JUCE交错格式
		for (int i = 0; i < fftSize; ++i)
		{
			fftBuffer[i * 2] = outre[i];
			fftBuffer[i * 2 + 1] = 0;
		}

		// 正向FFT
		fft->performRealOnlyForwardTransform(fftBuffer.data(), true);

		// 频域复数乘法
		for (int i = 0; i < fftSize; ++i)
		{
			float re = fftBuffer[i * 2];
			float im = fftBuffer[i * 2 + 1];

			fftBuffer[i * 2] = re * firre[i] - im * firim[i];
			fftBuffer[i * 2 + 1] = re * firim[i] + im * firre[i];
		}

		// 逆向FFT
		fft->performRealOnlyInverseTransform(fftBuffer.data());

		// 提取实部作为输出
		for (int i = 0; i < fftSize; i++)
		{
			outre[i] = fftBuffer[i * 2];
		}
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
			ComputeFFT();
		}
		return outre[posHop + firlen];
	}

	inline int GetLatencySamples()
	{
		return firlen;
	}

	inline void UpdatePosition(float x)
	{
		buffer[pos] = x;
		pos++;
		posHop++;
		if (pos >= fftSize) pos = 0;
		if (posHop >= firlen) posHop = 0;
	}

	inline float GetOutput()
	{
		return outre[posHop + firlen];
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

// 优化后的多线程版本
class LMConvolution1
{
private:
	constexpr static int firlen = 32;
	constexpr static int MaxStages = 256;
	ConvolutionFIR cfir;
	std::vector<ConvolutionFFT> cffts;
	std::vector<DelayLine> delays;
	int numStages = 0;

	// 多线程相关
	juce::ThreadPool threadPool;
	std::unique_ptr<std::atomic<bool>[]> stageReady;
	std::vector<bool> stageNeedsCompute;
	std::vector<float> delayedInputs;

	class FFTComputeJob : public juce::ThreadPoolJob
	{
	public:
		ConvolutionFFT* cfft;
		std::atomic<bool>* ready;

		FFTComputeJob(ConvolutionFFT* c, std::atomic<bool>* r)
			: ThreadPoolJob("FFTCompute"), cfft(c), ready(r)
		{
		}

		JobStatus runJob() override
		{
			cfft->ComputeFFT();
			ready->store(true, std::memory_order_release);
			return jobHasFinished;
		}
	};

public:
	LMConvolution1()
		: threadPool(std::thread::hardware_concurrency() - 1) // 保留一个核心给主线程
	{
		cffts.resize(MaxStages);
		delays.resize(MaxStages);
		stageReady = std::make_unique<std::atomic<bool>[]>(MaxStages);
		stageNeedsCompute.resize(MaxStages);
		delayedInputs.resize(MaxStages);

		for (int i = 0; i < MaxStages; ++i)
		{
			stageReady[i].store(true, std::memory_order_relaxed);
			stageNeedsCompute[i] = false;
		}
	}

	~LMConvolution1()
	{
		threadPool.removeAllJobs(true, 5000);
	}

	void SetConvolutionData(float* fir, int numSamples)
	{
		// 等待所有任务完成
		threadPool.removeAllJobs(true, 5000);

		numStages = 0;
		cfir.SetConvolutionData(fir, std::min(numSamples, firlen));
		if (numSamples <= firlen) return;

		numSamples -= firlen;
		for (int i = 0, pos = firlen; i < MaxStages; ++i)
		{
			int len = firlen << i;
			if (len > 16384) len = 16384;
			cffts[i].SetConvolutionData(&fir[pos], std::min(numSamples, len));
			delays[i].SetDelayTime(pos - cffts[i].GetLatencySamples());
			stageReady[i].store(true, std::memory_order_relaxed);
			stageNeedsCompute[i] = false;

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

		// 第一遍:更新位置并检查哪些stage需要FFT计算
		for (int i = 0; i < numStages; i++)
		{
			delayedInputs[i] = delays[i].ProcessSample(x);
			cffts[i].UpdatePosition(delayedInputs[i]);

			if (cffts[i].NeedFFTCompute())
			{
				stageNeedsCompute[i] = true;
				stageReady[i].store(false, std::memory_order_release);
			}
		}

		// 提交需要计算的FFT任务到线程池
		for (int i = 0; i < numStages; i++)
		{
			if (stageNeedsCompute[i])
			{
				threadPool.addJob(new FFTComputeJob(&cffts[i], &stageReady[i]), true);
				stageNeedsCompute[i] = false;
			}
		}

		// 等待所有计算完成并累加结果
		for (int i = 0; i < numStages; i++)
		{
			// 自旋等待(通常很快)
			while (!stageReady[i].load(std::memory_order_acquire))
			{
				std::this_thread::yield();
			}
			y += cffts[i].GetOutput();
		}

		return y;
	}

	// 批处理版本 - 更高效
	inline void ProcessBlock(const float* input, float* output, int numSamples)
	{
		for (int i = 0; i < numSamples; ++i)
		{
			output[i] = ProcessSample(input[i]);
		}
	}
};

class TestConvolution
{
public:
	constexpr static int TestLen = 131072 * 2;
private:
	LMConvolution1 convolution;
	float testdatre[TestLen];
	float testdatim[TestLen];

	int log2Int(int n)
	{
		int order = 0;
		while (n > 1)
		{
			n >>= 1;
			order++;
		}
		return order;
	}

public:
	TestConvolution()
	{
		for (int i = 0; i < TestLen / 2; ++i)
		{
			float x = (float)i / (TestLen / 2);
			x = x * x * TestLen / 4;
			testdatre[i] = cosf(x * 2.0f * (float)M_PI);
			testdatim[i] = -sinf(x * 2.0f * (float)M_PI);
		}
		for (int i = 0; i < 10; ++i)
		{
			float x = i / 10.0f;
			x = expf(x * 8.0f - 8.0f);
			testdatre[i] *= x;
			testdatim[i] *= x;
		}

		for (int i = TestLen / 2; i < TestLen; ++i)
		{
			testdatre[i] = 0;
			testdatim[i] = 0;
		}

		// 使用JUCE FFT
		int fftOrder = log2Int(TestLen);
		juce::dsp::FFT fft(fftOrder);

		std::vector<float> fftBuffer(TestLen * 2);
		for (int i = 0; i < TestLen; ++i)
		{
			fftBuffer[i * 2] = testdatre[i];
			fftBuffer[i * 2 + 1] = testdatim[i];
		}

		fft.performFrequencyOnlyForwardTransform(fftBuffer.data(), true);

		for (int i = 0; i < TestLen; ++i)
		{
			testdatre[i] = fftBuffer[i * 2] / (TestLen / 2);
			testdatim[i] = fftBuffer[i * 2 + 1] / (TestLen / 2);
		}

		convolution.SetConvolutionData(testdatre, TestLen);
	}

	float fbv = 0;
	void ProcessBlock(const float* in, float* out, int numSamples)
	{
		for (int i = 0; i < numSamples; i++)
		{
			out[i] = fbv = convolution.ProcessSample(in[i] - fbv * 0.90f);
		}
	}
};