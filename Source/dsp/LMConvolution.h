#pragma once

#define _USE_MATH_DEFINES
#include <future> 
#include <numeric> 
#include <math.h>
#include "fft.h"

class ConvolutionBase
{
private:
public:
};

class ConvolutionFIR :public ConvolutionBase
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

class ConvolutionFFT :public ConvolutionBase
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
class LMConvolution1 :public ConvolutionBase//优势:零延迟，非均匀切分ir
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
		cffts.resize(MaxStages);
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
			if (len > 16384)len = 16384;
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
	inline int GetLatencySamples()
	{
		return 0;
	}
};
class LMConvolution2 :public ConvolutionBase//优势:更新一次只需一次fft和ifft。延迟为块长度
{
private:
	std::vector<std::vector<std::complex<float>>> firs;
	std::vector<std::vector<std::complex<float>>> bufs;
	std::vector<float> blockbuf;
	std::vector<std::complex<float>> blockbuf_fft;  // 输入FFT专用
	std::vector<std::complex<float>> output_buf;    // 输出专用
	int posIn = 0, posHop = 0;
	int bufIndex = 0;  // 用于循环缓冲区索引，避免移位
	int numStages = 0;
	int blockSize = 4096;

public:
	LMConvolution2() {}

	void SetConvolutionData(float* fir, int numSamples, int blockSize = 4096)
	{
		this->blockSize = blockSize;
		blockbuf.resize(blockSize * 2, 0);
		blockbuf_fft.resize(blockSize * 2, { 0, 0 });
		output_buf.resize(blockSize * 2, { 0, 0 });

		firs.clear();
		bufs.clear();
		numStages = 0;
		bufIndex = 0;

		std::vector<std::complex<float>> tmp;
		int pos = 0;

		for (; pos < numSamples - blockSize; pos += blockSize, numStages++)
		{
			tmp.assign(blockSize * 2, { 0, 0 });
			for (int i = 0; i < blockSize; ++i)
			{
				tmp[i] = fir[i + pos];
			}
			fft_f32(tmp, blockSize * 2, 1);
			firs.push_back(tmp);

			// 预分配缓冲区
			bufs.push_back(std::vector<std::complex<float>>(blockSize * 2, { 0, 0 }));
		}

		if (numSamples - pos >= 1)
		{
			tmp.assign(blockSize * 2, { 0, 0 });
			for (int i = 0; i < numSamples - pos; ++i)
			{
				tmp[i] = fir[i + pos];
			}
			fft_f32(tmp, blockSize * 2, 1);
			firs.push_back(tmp);
			bufs.push_back(std::vector<std::complex<float>>(blockSize * 2, { 0, 0 }));
			numStages++;
		}
	}

	inline float ProcessSample(float x)
	{
		blockbuf[posIn] = x;
		posIn++;
		posHop++;
		if (posIn >= blockSize * 2) posIn = 0;

		if (posHop >= blockSize)
		{
			posHop = 0;

			// 复制输入块到FFT缓冲区
			if (posIn == 0)
			{
				for (int i = 0; i < blockSize * 2; i++)
					blockbuf_fft[i] = blockbuf[i];
			}
			else
			{
				for (int i = 0; i < blockSize; i++)
					blockbuf_fft[i] = blockbuf[i + blockSize];
				for (int i = 0; i < blockSize; i++)
					blockbuf_fft[i + blockSize] = blockbuf[i];
			}

			fft_f32(blockbuf_fft, blockSize * 2, 1);

			// 使用循环索引，避免移位
			bufIndex = (bufIndex - 1 + numStages) % numStages;
			bufs[bufIndex] = blockbuf_fft;

			// 累加卷积结果
			std::fill(output_buf.begin(), output_buf.end(), std::complex<float>(0, 0));
			for (int n = 0; n < numStages; ++n)
			{
				int idx = (bufIndex + n) % numStages;
				for (int i = 0; i < blockSize * 2; ++i)
				{
					output_buf[i] += firs[n][i] * bufs[idx][i];
				}
			}

			fft_f32(output_buf, blockSize * 2, -1);
		}

		// 从输出缓冲区读取，跳过前半部分(overlap)
		return output_buf[posHop + blockSize].real() / (blockSize * 2);
	}
	inline int GetBlockSize()
	{
		return blockSize;
	}
	inline int GetLatencySamples()
	{
		return blockSize;
	}

};
class LMConvolution2Async
{
private:
	LMConvolution2 conv;
	std::thread process_thread;
	std::atomic_bool should_stop{ false };

	// 无锁三缓冲区系统
	struct BufferSlot {
		std::vector<float> data;
		std::atomic<int> state{ 0 }; // 0=空闲, 1=填充中, 2=待处理, 3=处理中
	};

	BufferSlot buffers[3];
	std::atomic<int> current_fill{ 0 };    // 主线程当前填充的槽
	std::atomic<int> current_process{ -1 }; // 工作线程当前处理的槽
	std::atomic<int> current_output{ -1 };  // 主线程当前输出的槽

	int fill_pos = 0;
	int output_pos = 0;
	int block_size = 0;

	// 延迟补偿：预填充一块数据
	bool first_block_done = false;

public:
	~LMConvolution2Async()
	{
		should_stop.store(true);
		if (process_thread.joinable()) {
			process_thread.join();
		}
	}

	void SetConvolutionData(float* fir, int numSamples, int blockSize = 4096)
	{
		// 停止旧线程
		should_stop.store(true);
		if (process_thread.joinable()) {
			process_thread.join();
		}

		block_size = blockSize;

		// 初始化卷积核心
		conv.SetConvolutionData(fir, numSamples, block_size);

		// 初始化三个缓冲槽
		for (int i = 0; i < 3; ++i) {
			buffers[i].data.assign(block_size, 0);
			buffers[i].state.store(0);
		}

		fill_pos = 0;
		output_pos = 0;
		first_block_done = false;
		current_fill.store(0);
		current_process.store(-1);
		current_output.store(-1);
		should_stop.store(false);

		// 启动工作线程
		process_thread = std::thread([this]() {
			while (!should_stop.load(std::memory_order_relaxed)) {
				bool processed = false;

				// 查找待处理的槽
				for (int i = 0; i < 3; ++i) {
					int expected = 2; // 待处理状态
					if (buffers[i].state.compare_exchange_strong(
						expected, 3, std::memory_order_acquire)) {

						// 处理这个槽
						current_process.store(i, std::memory_order_release);

						for (int j = 0; j < block_size; ++j) {
							buffers[i].data[j] = conv.ProcessSample(buffers[i].data[j]);
						}

						// 标记为已完成（可输出）
						buffers[i].state.store(4, std::memory_order_release);
						processed = true;
						break;
					}
				}

				if (!processed) {
					// 没有待处理数据，短暂休眠
					std::this_thread::yield();
				}
			}
			});
	}

	inline float ProcessSample(float x)
	{
		int fill_slot = current_fill.load(std::memory_order_relaxed);

		// 填充当前槽
		buffers[fill_slot].data[fill_pos] = x;

		// 从输出槽读取数据
		float y = 0.0f;
		int output_slot = current_output.load(std::memory_order_acquire);

		if (output_slot >= 0) {
			// 有可用输出
			y = buffers[output_slot].data[output_pos];
			output_pos++;

			// 输出块完成
			if (output_pos >= block_size) {
				output_pos = 0;
				// 释放输出槽
				buffers[output_slot].state.store(0, std::memory_order_release);
				current_output.store(-1, std::memory_order_release);
			}
		}

		fill_pos++;

		// 填充块完成
		if (fill_pos >= block_size) {
			fill_pos = 0;

			// 标记当前槽为待处理
			buffers[fill_slot].state.store(2, std::memory_order_release);

			// 如果这是第一块，不要立即输出（保持blockSize延迟）
			if (!first_block_done) {
				first_block_done = true;
			}
			else {
				// 等待有已完成的槽可以输出
				for (int attempts = 0; attempts < 1000; ++attempts) {
					for (int i = 0; i < 3; ++i) {
						int expected = 4; // 已完成状态
						if (buffers[i].state.compare_exchange_strong(
							expected, 5, std::memory_order_acquire)) {
							current_output.store(i, std::memory_order_release);
							goto found_output;
						}
					}
					std::this_thread::yield();
				}
			found_output:;
			}

			// 查找下一个空闲槽用于填充
			for (int attempts = 0; attempts < 1000; ++attempts) {
				for (int i = 0; i < 3; ++i) {
					int expected = 0; // 空闲状态
					if (buffers[i].state.compare_exchange_strong(
						expected, 1, std::memory_order_acquire)) {
						current_fill.store(i, std::memory_order_release);
						goto found_fill;
					}
				}
				std::this_thread::yield();
			}
		found_fill:;
		}

		return y;
	}

	inline int GetLatencySamples()
	{
		return block_size;
	}
};

class LMConvolution3 :public ConvolutionBase//优势:结合1 2
{
private:
	constexpr static int firlen = 32;//must be power of 2
	constexpr static int MaxStages = 256;
	ConvolutionFIR cfir;
	std::vector<LMConvolution2Async> cffts;
	std::vector<DelayLine> delays;
	int numStages = 0;
public:
	LMConvolution3()
	{
		cffts.resize(MaxStages);
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
			//if (len > 16384)len = 16384;
			cffts[i].SetConvolutionData(&fir[pos], std::min(numSamples, len), len);
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
	inline int GetLatencySamples()
	{
		return 0;
	}

};


class TestConvolution
{
public:
	constexpr static int TestLen = 65536 * 32;
private:
	LMConvolution3 convolution;
	float testdatre[TestLen];
	float testdatim[TestLen];
public:
	TestConvolution()
	{
		for (int i = 0; i < TestLen / 2; ++i)
		{
			float x = (float)i / (TestLen / 2);
			x = x * x * TestLen / 4;
			testdatre[i] = cosf(x * 2.0 * M_PI);
			testdatim[i] = -sinf(x * 2.0 * M_PI);
		}
		for (int i = 0; i < 10; ++i)
		{
			float x = i / 10;
			x = expf(x * 8.0 - 8.0);
			testdatre[i] *= x;
			testdatim[i] *= x;
		}

		for (int i = TestLen / 2; i < TestLen; ++i)
		{
			testdatre[i] = 0;
			testdatim[i] = 0;
		}
		fft_f32(testdatre, testdatim, TestLen, 1);
		for (int i = 0; i < TestLen; ++i)
		{
			testdatre[i] /= TestLen / 2;
			testdatim[i] /= TestLen / 2;
		}
		convolution.SetConvolutionData(testdatre, TestLen);
	}
	float fbv = 0;
	void ProcessBlock(const float* in, float* out, int numSamples)
	{
		for (int i = 0; i < numSamples; i++)
		{
			out[i] = fbv = convolution.ProcessSample(in[i] - fbv * 0.90);
		}
	}
};