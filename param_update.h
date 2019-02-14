#pragma once

#include <memory>
#include <vector>
#include <stack>
#include <algorithm>
#include "word2vec.h"
#include "math_utils.h"
#include "huffman_tree.h"


template <typename Word>
class Syn1TrainStrategy
{
public:

	Syn1TrainStrategy(int layer1_size)
		: layer1_size_(layer1_size){}
	~Syn1TrainStrategy() {}

	virtual void train_syn1(const Word *current_word, const Vector& l1,  const float learning_rate, Vector& work) = 0;
	const int layer1_size_;
};


// Updateing context layer by Hierarchical Softmax
template <typename Word>
class HierarchicalSoftmax : public Syn1TrainStrategy<Word>
{
public:
	HierarchicalSoftmax(const int dim, const int layer1_size)
		: Syn1TrainStrategy<Word>(layer1_size) {
			// TODO: initialize syn1_ with value
			for (int i; i < layer1_size; ++i){

			}
		};
	~HierarchicalSoftmax() {};

	void build_tree(const std::vector<Word*>& words);
	virtual void train_syn1(const Word *current_word, const Vector& l1, const float learning_rate, Vector& work);

private:
	HuffmanTree<Word> encoder;
	std::vector<Vector> syn1_;
};

template <typename Word>
void HierarchicalSoftmax<Word>::build_tree(const std::vector<Word*>& words) {
	encoder.build_tree(words);
}

template <typename Word>
void HierarchicalSoftmax<Word>::train_syn1(const Word *current_word, const Vector& l1, const float learning_rate, Vector& work) {
	//if (current_word->codes_.empty()) // FIXME: does this really happen???
	//	return;

	// udate parameters by Hierarchical Softmax
	const auto code_length = encoder.encode(current_word).size();
	std::fill(work.begin(), work.end(), 0);
	for (size_t b = 0; b < code_length; ++b) {
		const int node_idx = encoder.encode(current_word)[b];
		auto& l2 = syn1_[node_idx];

		const float f = v::dot(l1, l2);
		if (cheap_math::is_out_of_exp_scale(f))
			return;

		const float g = (1 - encoder.encode(current_word)[b] - cheap_math::sigmoid(f)) * learning_rate;

		v::saxpy(work, g, l2); // work += syn1_[idx] * g;
		v::saxpy(l2, g, l1); // syn1_[idx] += syn0_[word_index] * g;
	}
}






// Updateing context layer by NegativeSampling
template <typename Word>
class NegativeSampling : public Syn1TrainStrategy<Word>
{
public:
	NegativeSampling();
	~NegativeSampling();

	virtual void train_syn1(const Word *word, const Vector& l1, const float learning_rate, Vector& work);
	virtual void get_grad_syn0(Vector& gradient);

private:
	int n_negative;
	std::vector<Vector> syn1neg_;
	std::vector<int> unigram_; // unigram frequenceies
};


template <typename Word>
void NegativeSampling<Word>::train_syn1(const Word *word, const Vector& l1, const float learning_rate, Vector& work) {
	for (int d = 0; d < n_negative + 1; ++d) {
		const int label = (d == 0? 1: 0);
		int target = 0;
		if (d == 0) target = i; // FIMXE: i = target position in the corpus
		else {
			target = unigram_[rand() % unigram_.size()];
			if (target == 0) target = rand() % (vocab_.size() - 1) + 1;
			if (target == i) continue;
		}

		auto& l2 = syn1neg_[target];
		const float f = v::dot(l1, l2);
		float g = 0;
		if (f > max_exp) g = (label - 1) * learning_rate;
		else if (f < -max_exp) g = (label - 0) * learning_rate;
		else {
			// int fi = int((f + max_exp) * (max_size / max_exp / 2.0));
			// g = (label - table[fi]) * learning_rate;
			g = (label - cheap_math::sigmoid(f)) * learning_rate;
		}

		v::saxpy(work, g, l2); // work += syn1_[idx] * g
		v::saxpy(l2, g, l1); // syn1_[idx] += syn0_[word_index] * g;
	}
}


