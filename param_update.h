#pragma once

#include <memory>
#include <vector>
#include <stack>
#include <algorithm>
#include <random>
#include <boost/utility.hpp>
#include "math_utils.h"
#include "huffman_tree.h"
#include "v.h"


template <typename Word>
struct Syn1TrainStrategy : private boost::noncopyable
{

	Syn1TrainStrategy(const int layer1_size, const int n_words)
		: layer1_size_(layer1_size), n_words_(n_words)
	{
		std::cout << "Initializing syn1_..." << std::endl;
		syn1_.resize(n_words);
		for (auto& s: syn1_) s.resize(layer1_size);
	}
	~Syn1TrainStrategy() {}

	virtual void train_syn1(const Word *current_word, const Vector& l1,  const float learning_rate, Vector& work) = 0;

	const int layer1_size_;
	const int n_words_;
	std::vector<Vector> syn1_;
};


// Updateing context layer by Hierarchical Softmax
template <typename Word>
class HierarchicalSoftmax : public Syn1TrainStrategy<Word>
{
public:
	HierarchicalSoftmax(const int layer1_size, const int n_words)
		: Syn1TrainStrategy<Word>(layer1_size, n_words) {
		};
	~HierarchicalSoftmax() {};

	void build_tree(const std::vector<Word*>& words);
	virtual void train_syn1(const Word *current_word, const Vector& l1, const float learning_rate, Vector& work) override;

private:
	HuffmanTree<Word> encoder;
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
	const auto& codes = encoder.encode(current_word);
	const auto& points = encoder.get_points(current_word);
	const auto code_length = codes.size();
	for (size_t b = 0; b < code_length; ++b) {
		const int node_idx = points[b];
		auto& l2 = this->syn1_[node_idx];

		const float f = v::dot(l1, l2);
		const float g = (1 - codes[b] - cheap_math::sigmoid(f)) * learning_rate;

		v::saxpy(work, g, l2); // work += syn1_[idx] * g;
		v::saxpy(l2, g, l1); // syn1_[idx] += syn0_[word_index] * g;
	}
}


// Updateing context layer by NegativeSampling
template <typename Word>
class NegativeSampling : public Syn1TrainStrategy<Word>
{
public:
	NegativeSampling(const int layer1_size, const int n_words, const int n_negative, const float power = 0.75)
		: Syn1TrainStrategy<Word>(layer1_size, n_words), n_negative_(n_negative), power_(power){}
	~NegativeSampling() = default;

	void update_distribution(const std::vector<Word *>& words_) {
		std::vector<double> unigram_prob;
		unigram_prob.reserve(words_.size());
		for (const auto& word : words_){
			unigram_prob.emplace_back(std::pow(word->count_, power_));
		}
		distribution = std::discrete_distribution<size_t>(unigram_prob.begin(), unigram_prob.end());
	}
	virtual void train_syn1(const Word *current_word, const Vector& l1, const float learning_rate, Vector& work);

private:
	const int n_negative_;
	const float power_;
	std::discrete_distribution<size_t> distribution;
};


template <typename Word>
void NegativeSampling<Word>::train_syn1(const Word *current_word, const Vector& l1, const float learning_rate, Vector& work) {
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	const auto& pred_word_idx = current_word->index_;
	for (int d = 0; d < n_negative_ + 1; ++d) { // a predicted contex word & negative words
		const int label = (d == 0? 1: 0);
		size_t target = 0;
		if (d == 0) {
			target = pred_word_idx;
		}
		else { // negative sampling
			target = distribution(engine);
			// if (target == 0) target = rand() % (vocab_.size() - 1) + 1;  //FIXME
			if (target == pred_word_idx) continue;
		}

		auto& l2 = this->syn1_[target];
		const float f = v::dot(l1, l2);
		const float g = (label - cheap_math::sigmoid(f)) * learning_rate;

		v::saxpy(work, g, l2); // work += syn1_[idx] * g
		v::saxpy(l2, g, l1); // syn1_[idx] += syn0_[word_index] * g;
	}
}


