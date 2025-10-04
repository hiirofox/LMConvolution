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

	// 双缓冲结构
	struct BufferSlot {
		std::vector<float> input;
		std::vector<float> output;
		std::atomic<int> state{ 0 };  // 0=空闲, 1=已填充待处理, 2=处理完成可输出
	};

	BufferSlot buffers[2];

	int fill_buffer = 0;      // 主线程当前填充的buffer
	int output_buffer = -1;   // 主线程当前输出的buffer

	int fill_pos = 0;
	int half_block_size = 0;  // 实际使用的块大小（用户指定的一半）

public:
	~LMConvolution2Async()
	{
		should_stop.store(true);
		if (process_thread.joinable()) {
			process_thread.join();
		}
	}

	void SetConvolutionData(float* fir, int numSamples, int blockSize = 32768)
	{
		should_stop.store(true);
		if (process_thread.joinable()) {
			process_thread.join();
		}

		// 关键：使用 blockSize/2 作为实际块大小
		half_block_size = blockSize / 2;

		// 卷积核心也使用 blockSize/2
		conv.SetConvolutionData(fir, numSamples, half_block_size);

		// 初始化双缓冲
		for (int i = 0; i < 2; ++i) {
			buffers[i].input.assign(half_block_size, 0);
			buffers[i].output.assign(half_block_size, 0);
			buffers[i].state.store(0);
		}

		fill_buffer = 0;
		output_buffer = -1;
		fill_pos = 0;
		should_stop.store(false);

		// 启动处理线程
		process_thread = std::thread([this]() {
			while (!should_stop.load(std::memory_order_relaxed)) {
				bool processed = false;

				// 查找待处理的buffer
				for (int i = 0; i < 2; ++i) {
					int expected = 1;
					if (buffers[i].state.compare_exchange_strong(
						expected, 2, std::memory_order_acquire)) {

						// 处理整个块
						for (int j = 0; j < half_block_size; ++j) {
							buffers[i].output[j] = conv.ProcessSample(buffers[i].input[j]);
						}

						// 保持 state=2，等待主线程使用
						processed = true;
						break;
					}
				}

				if (!processed) {
					//std::this_thread::yield();
					std::this_thread::sleep_for(std::chrono::microseconds(1));
				}
			}
			});
	}

	inline float ProcessSample(float x)
	{
		// 填充当前buffer
		buffers[fill_buffer].input[fill_pos] = x;

		// 从输出buffer读取
		float y = 0.0f;
		if (output_buffer >= 0) {
			y = buffers[output_buffer].output[fill_pos];
		}

		fill_pos++;

		// 半块填充完成
		if (fill_pos >= half_block_size) {
			fill_pos = 0;

			// 标记当前填充buffer为待处理
			buffers[fill_buffer].state.store(1, std::memory_order_release);

			// 切换到另一个buffer
			int next_fill = 1 - fill_buffer;

			// 等待下一个buffer准备好（状态为0或2）
			while (true) {
				int expected = 2;  // 先尝试处理完成状态
				if (buffers[next_fill].state.compare_exchange_strong(
					expected, 0, std::memory_order_acquire)) {
					// 处理完成的buffer，设为输出，并释放
					output_buffer = fill_buffer;
					break;
				}

				expected = 0;  // 再尝试空闲状态（第一次循环时）
				if (buffers[next_fill].state.compare_exchange_strong(
					expected, 0, std::memory_order_acquire)) {
					// 空闲buffer，不更新输出（第一次迭代）
					break;
				}

				// 都失败了，说明buffer还在处理中，等待
				//std::this_thread::yield();
				std::this_thread::sleep_for(std::chrono::microseconds(1));
			}

			fill_buffer = next_fill;
		}

		return y;
	}

	inline int GetLatencySamples()
	{
		// 总延迟 = half_block_size（卷积延迟） + half_block_size（缓冲延迟）
		return half_block_size * 2;
	}
};

class LMConvolution3 :public ConvolutionBase//优势:结合1 2
{
private:
	constexpr static int firlen = 256;//must be power of 2
	constexpr static int MaxStages = 256;
	ConvolutionFIR cfir;
	std::vector<std::unique_ptr<LMConvolution2Async>> cffts;
	std::vector<DelayLine> delays;
	int numStages = 0;
public:
	LMConvolution3()
	{
		cffts.reserve(MaxStages);  // 预留空间，避免重新分配
		for (int i = 0; i < MaxStages; ++i)
		{
			cffts.push_back(std::make_unique<LMConvolution2Async>());
		}
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
			cffts[i]->SetConvolutionData(&fir[pos], std::min(numSamples, len), len / 2);
			delays[i].SetDelayTime(pos - cffts[i]->GetLatencySamples());
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
			y += cffts[i]->ProcessSample(delays[i].ProcessSample(x));
		}
		return y;
	}

	inline int GetLatencySamples()
	{
		return 0;
	}
};

class LMConvolution4
{
private:
	constexpr static int firlen = 32;//must be power of 2
	constexpr static int MaxStages = 256;
	ConvolutionFIR cfir;
	std::vector<std::unique_ptr<LMConvolution2>> cffts;
	std::vector<std::unique_ptr<LMConvolution2Async>> cfftAsyncs;//块长度大于某值时开始使用
	std::vector<DelayLine> delays;
	int numStages = 0, numAsyncStages = 0;
	constexpr static int MinAsyncBlockSize = 2048;
public:
	LMConvolution4()
	{
		cffts.reserve(MaxStages);  // 预留空间，避免重新分配
		for (int i = 0; i < MaxStages; ++i)
		{
			cffts.push_back(std::make_unique<LMConvolution2>());
			cfftAsyncs.push_back(std::make_unique<LMConvolution2Async>());
		}
		delays.resize(MaxStages);
	}
	void SetConvolutionData(float* fir, int numSamples)
	{
		numStages = 0;
		numAsyncStages = 0;
		cfir.SetConvolutionData(fir, std::min(numSamples, firlen));
		if (numSamples <= firlen)return;
		numSamples -= firlen;
		for (int i = 0, pos = firlen; i < MaxStages; ++i)
		{
			int len = firlen << i;
			//if (len > 16384)len = 16384;
			if (len >= MinAsyncBlockSize)
			{
				cfftAsyncs[numAsyncStages]->SetConvolutionData(&fir[pos], std::min(numSamples, len), len);
				delays[numStages].SetDelayTime(pos - cfftAsyncs[numAsyncStages]->GetLatencySamples());
				numAsyncStages++;
			}
			else
			{
				cffts[numStages]->SetConvolutionData(&fir[pos], std::min(numSamples, len), len);
				delays[numStages].SetDelayTime(pos - cffts[numStages]->GetLatencySamples());
				numStages++;
			}
			if (numSamples <= len)
			{
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
			y += cffts[i]->ProcessSample(x);
		}
		for (int i = 0; i < numAsyncStages; i++)
		{
			y += cfftAsyncs[i]->ProcessSample(x);
		}
		return y;
	}
};

class TestConvolution
{
public:
	constexpr static int TestLen = 65536 * 512;
private:
	LMConvolution4 convolution;
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
			out[i] = fbv = convolution.ProcessSample(in[i] - fbv * 0.00);
		}
	}
};