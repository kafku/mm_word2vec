#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <stack>
#include <algorithm>
#include <random>
#include <boost/utility.hpp>
#include "math_utils.h"
#include "huffman_tree.h"
#include "v.h"
#include "cvt.h"
#include "model_generated.h"


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




template <typename Word>
struct Syn0TrainStrategy : private boost::noncopyable
{

	Syn0TrainStrategy(const int layer1_size, const int n_words)
		: layer1_size_(layer1_size), n_words_(n_words)
	{
		std::cout << "Initializing syn0_..." << std::endl;
		std::random_device seed_gen;
		std::default_random_engine eng(seed_gen());
		std::uniform_real_distribution<float> rng(0.0, 1.0);
		syn0_.resize(n_words);
		for (auto& s: syn0_) {
			s.resize(layer1_size);
			for (auto& x: s) x = (rng(eng) - 0.5) / layer1_size;
		}
	}
	Syn0TrainStrategy(std::vector<Vector>&& src)
		: layer1_size_(src[0].size()), n_words_(src.size()), syn0_(src) {}
	~Syn0TrainStrategy() {}

	virtual void train_syn0(const Word *current_word, const Vector& work, const float learning_rate) = 0;

	const int layer1_size_;
	const int n_words_;
	std::vector<Vector> syn0_;
};


// simple gradient descent
template <typename Word>
class SimpleGD : public Syn0TrainStrategy<Word>
{
public:
	SimpleGD(const int layer1_size, const int n_words)
		: Syn0TrainStrategy<Word>(layer1_size, n_words) {}

	virtual void train_syn0(const Word *current_word, const Vector& work, const float learning_rate) override;
};

template<typename Word>
void SimpleGD<Word>::train_syn0(const Word *current_word, const Vector& work, const float learning_rate) {
	auto& l1 = this->syn0_[current_word->index_];
	v::saxpy(l1, 1.0, work); // syn0_[current_word->index] += work;
}



// jointlty learn word vectors and a linear transformation to other modality
template <typename Word, typename Func>
class MultimodalGD : public Syn0TrainStrategy<Word>
{
public:
	using String = decltype(std::declval<Word>().text_);

	MultimodalGD(const int layer1_size, const int n_words, const int n_negative, const float margin, const float reg_param, const uint32_t min_freq=500)
		: Syn0TrainStrategy<Word>(layer1_size, n_words), n_negative_(n_negative), margin_(margin), reg_param_(reg_param), min_freq_(min_freq)
  {
    std::cout << "  margin: " << margin << std::endl;
    std::cout << "  regularization param: " << reg_param << std::endl;
    std::cout << "  negative samples: " << n_negative << std::endl;
    std::cout << "  minimum freq.: " << min_freq << std::endl;
  }

	virtual void load(const std::string& file);
	virtual void train_syn0(const Word *current_word, const Vector& work, const float learning_rate) override;

private:
	const float margin_;
	const float reg_param_; // regularization parameter
	const int n_negative_;
	const uint32_t min_freq_;
	v::LightMatrix linear_transform;
	Vector linear_transform_vec;
	std::vector<Vector> multimodal_data;
	std::unordered_map<String, size_t> vocab2data_idx;
	std::uniform_int_distribution<size_t> distribution;
};

template <typename Word, typename Func>
void MultimodalGD<Word, Func>::load(const std::string& file) {
  std::cout << "Loading multimodal features from " << file << std::endl;
	// load serialized data
	std::ifstream in(file, std::ifstream::binary);
	std::stringstream ss;
	ss << in.rdbuf();
	std::string s = ss.str();

	const word2vec::Dict *dict = word2vec::GetDict(s.data());
	const size_t n_words = dict->words()->Length();

	multimodal_data.clear();
	multimodal_data.resize(n_words);

	for (size_t i = 0; i < n_words; ++i) {
		const auto *word = dict->words()->Get(i);
		auto name = Cvt<String>::from_utf8(word->name()->c_str());
		auto p = vocab2data_idx.emplace(name, i);
		multimodal_data[i] = std::vector<float>{word->feature()->begin(), word->feature()->end()};
	}
  std::cout << "  Loaded " << vocab2data_idx.size() << " multimodal features" << std::endl;

	// initialize linear transformation matrix
	const auto n_rows = multimodal_data[0].size();
	const auto n_cols = this->layer1_size_;

	linear_transform_vec.resize(n_words * n_cols);
	std::random_device seed_gen;
	std::default_random_engine eng(seed_gen());
	std::uniform_real_distribution<float> rng(0.0, 1.0); // FIXME: is this OK?
	for (auto& x: linear_transform_vec) {
		x = (rng(eng) - 0.5) / (n_rows * n_cols);
	}

	linear_transform = v::LightMatrix(n_rows, n_cols,
			linear_transform_vec.data(), linear_transform_vec.data() + linear_transform_vec.size());
  std::cout << "  Constructed " << n_rows << " x " << n_cols << " matrix" << std::endl;

	distribution = std::uniform_int_distribution<size_t>(1, n_words - 1);
}

template<typename Word, typename Func>
void MultimodalGD<Word, Func>::train_syn0(const Word *current_word, const Vector& work, const float learning_rate) {
	auto& l1 = this->syn0_[current_word->index_];
	v::saxpy(l1, 1.0, work); // syn0_[current_word->index] += work;

	// do nothing if the word is not in vocab
	auto it = vocab2data_idx.find(current_word->text_);
	if (it == vocab2data_idx.end())
		return;

	// skip less frequent words
	if (current_word->count_ < min_freq_)
		return;

	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	const auto idx = it->second;
	const auto& pos_vec = multimodal_data[idx];

	// negative sampling
	std::vector<size_t> neg_indices(n_negative_);
	for (auto& x : neg_indices) {
		x = (idx + distribution(engine)) % multimodal_data.size();
	}

	Vector grad_syn0(this->layer1_size_, 0);
	Vector grad_lin_trans_vec(linear_transform.size(), 0);
	v::LightMatrix grad_lin_trans(linear_transform.n_rows, linear_transform.n_cols,
			grad_lin_trans_vec.data(), grad_lin_trans_vec.data() + grad_lin_trans_vec.size());

	int n_violate = 0;
	// update matrix
	#pragma omp critical
	{
		for (const auto neg_idx : neg_indices) {
			const auto& neg_vec = multimodal_data[neg_idx];
			const float diff = Func::call(pos_vec, l1, linear_transform) - Func::call(neg_vec, l1, linear_transform);
			if (margin_ <= diff) continue;

			// update gradient for syn0
			Func::grad_wrt_vec(learning_rate, l1, pos_vec, linear_transform, grad_syn0);
			Func::grad_wrt_vec(-learning_rate, l1, neg_vec, linear_transform, grad_syn0);

			// update gradient for linear_transform
			Func::grad_wrt_mat(learning_rate, l1, pos_vec, linear_transform, grad_lin_trans);
			Func::grad_wrt_mat(-learning_rate, l1, neg_vec, linear_transform, grad_lin_trans);

			++n_violate;
		}
	}
	if (n_violate == 0) return; // do nothing when no violations

	v::saxpy(l1, 1.0, grad_syn0); // update syn0_
	v::saxpy(linear_transform, 1.0, grad_lin_trans); // update linear_transform
	v::saxpy(linear_transform, -learning_rate * reg_param_, linear_transform); // regularization
}
