// Word2Vec C++ 11
// Copyright (C) 2013 Jack Deng <jackdeng@gmail.com>
// MIT License
//
// based on Google Word2Vec and gensim from Radim Rehurek (see http://radimrehurek.com/2013/09/deep-learning-with-word2vec-and-gensim/)
// see main.cc for building instructions
// P.S. compiler vector extension, AVX intrinsics (even with FMA) doesn't seem to help compared to compilation with -Ofast -march=native on a haswell laptop

#include <vector>
#include <list>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <fstream>
#include <sstream>

#include <chrono>
#include <cstdio>

#include "v.h"
#include "param_update.h"

#include "model_generated.h"

template <typename T> struct Cvt;

template <> struct Cvt<std::string> {
	static const std::string& to_utf8(const std::string& s) { return s; }
	static const std::string& from_utf8(const std::string& s) { return s; }
};

#if defined(_LIBCPP_BEGIN_NAMESPACE_STD)
#include <codecvt>
template <> struct Cvt<std::u16string> {
	static std::string to_utf8(const std::u16string& in) {
	    std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> cv;
    	return cv.to_bytes(in.data());
	}

	static std::u16string from_utf8(const std::string& in) {
	    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> cv;
    	return cv.from_bytes(in.data());
	}
};
#else // gcc has no <codecvt>
#include "utf8cpp/utf8.h"
template <> struct Cvt<std::u16string> {
	static std::string to_utf8(const std::u16string& in) {
		std::string out;
		utf8::utf16to8(in.begin(), in.end(), std::back_inserter(out));
		return out;
	}

	static std::u16string from_utf8(const std::string& in) {
		std::u16string out;
		utf8::utf8to16(in.begin(), in.end(), std::back_inserter(out));
		return out;
	}
};
#endif

template <class String = std::string>
struct Word2Vec
{
	enum Tag { S = 0, B, M, E };
	static const char *tag_string(Tag t) {
		switch(t) {
		case S: return "S";
		case B: return "B";
		case M: return "M";
		case E: return "E";
		}
	}

	struct Word
	{
		int32_t index_;
		String text_;
		uint32_t count_;

		Word(int32_t index, String text, uint32_t count) : index_(index), text_(text), count_(count) {}
		Word(const Word&) = delete;
		const Word& operator = (const Word&) = delete;
	};
	typedef std::shared_ptr<Word> WordP;

	struct Sentence
	{
		std::vector<Word *> words_;
		std::vector<String> tokens_;
		std::vector<Tag> tags_;
	};
	typedef std::shared_ptr<Sentence> SentenceP;

	std::vector<Vector> syn0_;
	std::vector<Vector> syn0norm_;

	std::unordered_map<String, WordP> vocab_;
	std::vector<Word *> words_;

	const int layer1_size_; // # of units (embedding dim)
	const int window_; // window size

	const float sample_; // subsampling

	const int min_count_; // minimum frequency of word
	const int negative_; // # of negative samples

	const float alpha_, min_alpha_; // learning rate

	const bool phrase_;
	const float phrase_threshold_;


	Word2Vec(int size = 100, int window = 5, float sample = 0.001, int min_count = 5, const std::string& train_method = "HS", int negative = 0, float alpha = 0.025, float min_alpha = 0.0001)
		:layer1_size_(size), window_(window), sample_(sample), min_count_(min_count), syn1_train_method_(train_method), negative_(negative),
		alpha_(alpha), min_alpha_(min_alpha), phrase_(false), phrase_threshold_(100) {}

	bool has(const String& w) const { return vocab_.find(w) != vocab_.end(); }

	int build_vocab(std::vector<SentenceP>& sentences) {
		size_t count = 0;
		std::unordered_map<String, int> vocab;
		auto progress = [&count](const char *type, const std::unordered_map<String, int>& vocab) {
			printf("collecting [%s] %lu sentences, %lu distinct %ss, %d %ss\n", type, count, vocab.size(), type,
				std::accumulate(vocab.begin(), vocab.end(), 0, [](int x, const std::pair<String, int>& v) { return x + v.second; }), type);
		};

		for (auto& sentence: sentences) {
			++count;
			if (count % 10000 == 0) progress("word", vocab);

			String last_token;
			for (auto& token: sentence->tokens_) {
				vocab[token] += 1;
				// add bigram phrases
				if (phrase_) {
					if(!last_token.empty()) vocab[last_token + Cvt<String>::from_utf8("_") + token] += 1;
					last_token = token;
				}
			}
		}
		progress("word", vocab);

		if (phrase_) { // build phrase
			count = 0;
			int total_words = std::accumulate(vocab.begin(), vocab.end(), 0, [](int x, const std::pair<String, int>& v) { return x + v.second; });

			std::unordered_map<String, int> phrase_vocab;

			for (auto& sentence: sentences) {
				++count;
				if (count % 10000 == 0) progress("phrase", phrase_vocab);

				std::vector<String> phrase_tokens;
				String last_token;
				uint32_t pa = 0, pb = 0, pab = 0;
				for (auto& token: sentence->tokens_) {
					pb = vocab[token];
					if (!	last_token.empty()) {
						String phrase = last_token + Cvt<String>::from_utf8("_") + token;
						pab = vocab[phrase];
						float score = 0;
						if (pa >= min_count_ && pb >= min_count_ && pab >= min_count_)
							score = (pab - min_count_ ) / (float(pa) * pb) * total_words;
						if (score > phrase_threshold_) {
							phrase_tokens.push_back(phrase);
							token.clear();
							phrase_vocab[phrase] += 1;
						}
						else {
							phrase_tokens.push_back(last_token);
							phrase_vocab[last_token] += 1;
						}
					}
					last_token = token;
					pa = pb;
				}

				if (!last_token.empty()) {
					phrase_tokens.push_back(last_token);
					phrase_vocab[last_token] += 1;
				}
				sentence->tokens_.swap(phrase_tokens);
			}
			progress("phrase", phrase_vocab);

			printf("using phrases\n");
			vocab.swap(phrase_vocab);
		} // build phrase

		int n_words = vocab.size();
		if (n_words <= 1) return -1;

		words_.reserve(n_words);

		for (auto& p: vocab) {
			uint32_t count = p.second;
			if (count <= min_count_) continue;

			auto r = vocab_.emplace(p.first, WordP(new Word{0, p.first, count}));
			words_.push_back((r.first->second.get()));
		}
		std::sort(words_.begin(), words_.end(),
				[](Word *w1, Word *w2) { return w1->count_ > w2->count_; });

		int index = 0;
		for (auto& w: words_) w->index_ = index++;

		printf("collected %lu distinct words with min_count=%d\n", vocab_.size(), min_count_);

		n_words = words_.size();
		if (syn1_train_method_ == "HS") { // Hierarchical Softmax
			std::shared_ptr<HierarchicalSoftmax<Word>> HS_strategy = std::make_shared<HierarchicalSoftmax<Word>>(layer1_size_, n_words);
			HS_strategy->build_tree(words_);
			syn1_train_ = HS_strategy;
		}
		else if (syn1_train_method_ == "NS") { // Negative Sampling
			std::shared_ptr<NegativeSampling<Word>> NS_strategy = std::make_shared<NegativeSampling<Word>>(layer1_size_, n_words, negative_);
			NS_strategy->build_tree(words_);
			syn1_train_ = NS_strategy;
		}
		else {
			// NOTE: invalid option
			// TODO: show error message
			return -1;
		}

		// initialize syn0_
		std::default_random_engine eng(::time(NULL));
		std::uniform_real_distribution<float> rng(0.0, 1.0);
		syn0_.resize(n_words);
		for (auto& s: syn0_) {
			s.resize(layer1_size_);
			for (auto& x: s) x = (rng(eng) - 0.5) / layer1_size_;
		}

		return 0;
	}

	int train(std::vector<SentenceP>& sentences, int n_workers) {
		int total_words = std::accumulate(vocab_.begin(), vocab_.end(), 0,
			[](int x, const std::pair<String, WordP>& p) { return (int)(x + p.second->count_); });
		int current_words = 0;
		const float alpha0 = alpha_, min_alpha = min_alpha_;

		std::default_random_engine eng(::time(NULL));
		std::uniform_real_distribution<float> rng(0.0, 1.0);

		size_t n_sentences = sentences.size();

		size_t last_words = 0;
		auto cstart = std::chrono::high_resolution_clock::now();

		printf("training %d sentences\n", n_sentences);

		#pragma omp parallel for default(shared)
		for (size_t i = 0; i < n_sentences; ++i) {
			auto sentence = sentences[i].get();
			if (sentence->tokens_.empty())
				continue;
			size_t len = sentence->tokens_.size();
			for (size_t i=0; i<len; ++i) {
				auto it = vocab_.find(sentence->tokens_[i]);
				if (it == vocab_.end()) continue;
				Word *word = it->second.get();
				// subsampling
				if (sample_ > 0) {
					float rnd = (sqrt(word->count_ / (sample_ * total_words)) + 1) * (sample_ * total_words) / word->count_;
					if (rnd < rng(eng)) continue;
				}
				sentence->words_.emplace_back(it->second.get());
			}

			float alpha = std::max(min_alpha, float(alpha0 * (1.0 - 1.0 * current_words / total_words)));
			size_t words = train_sentence(*sentence, alpha);

			#pragma omp atomic
			current_words += words;

			if (current_words - last_words > 1024 * 100 || i == n_sentences - 1) {
				auto cend = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count();
				printf("training alpha: %.4f progress: %.2f%% words per sec: %.3fK\n", alpha, current_words * 100.0/total_words, (current_words - last_words) * 1000.0 / duration);
				last_words = current_words;
				cstart = cend;
			}
		}

		syn0norm_ = syn0_;
		for (auto& v: syn0norm_) v::unit(v);

		return 0;
	}

	int save(const std::string& file) const {
		flatbuffers::FlatBufferBuilder fbb;

		std::vector<Word *> words = words_;
		std::sort(words.begin(), words.end(), [](Word *w1, Word *w2) { return w1->count_ > w2->count_; });

		std::vector<flatbuffers::Offset<word2vec::Word>> ws;
		for (auto w: words) {
			auto name = fbb.CreateString(Cvt<String>::to_utf8(w->text_));
			ws.push_back(word2vec::CreateWord(fbb, name, fbb.CreateVector(syn0_[w->index_])));
		}

		auto dict = word2vec::CreateDict(fbb, fbb.CreateVector(ws.data(), ws.size()));
		fbb.Finish(dict);

		std::ofstream out(file, std::ofstream::out | std::ofstream::binary);
		out.write((const char *)fbb.GetBufferPointer(), fbb.GetSize());
		return 0;
	}

	int save_text(const std::string& file) const {
		std::ofstream out(file, std::ofstream::out);
		out << syn0_.size() << " " << syn0_[0].size() << std::endl;

		std::vector<Word *> words = words_;
		std::sort(words.begin(), words.end(), [](Word *w1, Word *w2) { return w1->count_ > w2->count_; });

		for (auto w: words) {
			out << Cvt<String>::to_utf8(w->text_);
			for (auto i: syn0_[w->index_]) out << " " << i;
			out << std::endl;
		}

		return 0;
	}

	int load(const std::string& file) {
		std::ifstream in(file, std::ifstream::binary);
		std::stringstream ss;
		ss << in.rdbuf();
		std::string s = ss.str();

		const word2vec::Dict *dict = word2vec::GetDict(s.data());
		size_t n_words = dict->words()->Length();

		syn0_.clear(); vocab_.clear(); words_.clear();
		syn0_.resize(n_words);

		for (int i=0; i<n_words; ++i) {
			const auto *word = dict->words()->Get(i);
			auto name = Cvt<String>::from_utf8(word->name()->c_str());
			auto p = vocab_.emplace(name, std::make_shared<Word>(i, name, 0));
			words_.push_back(p.first->second.get());
			syn0_[i] = std::vector<float>{word->feature()->begin(), word->feature()->end()};
		}

		layer1_size_ = syn0_[0].size();
		printf("%d words loaded\n", n_words);

		syn0norm_ = syn0_;
		for (auto& v: syn0norm_) v::unit(v);

		return 0;
	}

	int load_text(const std::string& file) {
		std::ifstream in(file);
		std::string line;
		if (! std::getline(in, line)) return -1;

		int n_words = 0, layer1_size = 0;
		std::istringstream iss(line);
		iss >> n_words >> layer1_size;

		syn0_.clear(); vocab_.clear(); words_.clear();
		syn0_.resize(n_words);
		for (int i=0; i<n_words; ++i) {
			if (! std::getline(in, line)) return -1;

			std::istringstream iss(line);
			std::string text;
			iss >> text;

			auto p = vocab_.emplace(Cvt<String>::from_utf8(text), WordP(new Word{i, Cvt<String>::from_utf8(text), 0}));
			words_.push_back(p.first->second.get());
			syn0_[i].resize(layer1_size);
			for(int j=0; j<layer1_size; ++j) {
				iss >> syn0_[i][j];
			}
		}

		layer1_size_ = layer1_size;
		printf("%d words loaded\n", n_words);

		syn0norm_ = syn0_;
		for (auto& v: syn0norm_) v::unit(v);

		return 0;
	}

	const Vector& word_vector(const String& w) const {
		static Vector nil;
		auto it = vocab_.find(w);
		if (it == vocab_.end()) return nil;
		return syn0_[it->second->index_];
	}

	size_t word_vector_size() const { return layer1_size_; }

	std::vector<std::pair<String,float>> most_similar(std::vector<String> positive, std::vector<String> negative, int topn) {
		if ((positive.empty() && negative.empty()) || syn0norm_.empty()) return std::vector<std::pair<String,float>>{};

		Vector mean(layer1_size_);
		std::vector<int> all_words;
		auto add_word = [&mean, &all_words, this](const String& w, float weight) {
			auto it = vocab_.find(w);
			if (it == vocab_.end()) return;

			Word& word = *it->second;
			v::saxpy(mean, weight, syn0norm_[word.index_]);

			all_words.push_back(word.index_);
		};

		for (auto& w: positive) add_word(w, 1.0);
		for (auto& w: negative) add_word(w, -1.0);

		v::unit(mean);

		Vector dists;
		std::vector<int> indexes;
		int i=0;

		dists.reserve(syn0norm_.size());
		indexes.reserve(syn0norm_.size());
		for (auto &x: syn0norm_) {
			dists.push_back(v::dot(x, mean));
			indexes.push_back(i++);
		}

		auto comp = [&dists](int i, int j) { return dists[i] > dists[j]; };
//		std::sort(indexes.begin(), indexes.end(), comp);

		int k = std::min(int(topn+all_words.size()), int(indexes.size())-1);
		auto first = indexes.begin(), last = indexes.begin() + k, end = indexes.end();
		std::make_heap(first, last + 1, comp);
		std::pop_heap(first, last + 1, comp);
		for (auto it = last + 1; it != end; ++it) {
			if (! comp(*it, *first)) continue;
		    	*last = *it;
		    	std::pop_heap(first, last + 1, comp);
		}

		std::sort_heap(first, last, comp);

		std::vector<std::pair<String,float>> results;
		for(int i=0, j=0; i<k; ++i) {
			if (std::find(all_words.begin(), all_words.end(), indexes[i]) != all_words.end())
				continue;
			results.push_back(std::make_pair(words_[indexes[i]]->text_, dists[indexes[i]]));
			if (++j >= topn) break;
		}

		return results;
	}

private:
	int train_sentence(const Sentence& sentence, const float alpha) {
		Vector grad_syn0(layer1_size_);

		int count = 0;
		const int sentence_length = sentence.words_.size();
		const int reduced_window = rand() % window_;
		for (int i = 0; i < sentence_length; ++i) { // loop: tokens in corpus
			const Word& current_word = *sentence.words_[i];

			int j = std::max(0, i - window_ + reduced_window);
			int k = std::min(sentence_length, i + window_ + 1 - reduced_window);
			for (; j < k; ++j) { // loop: window
				if (j == i) continue;
				const Word *word = sentence.words_[j]; // predicted-word index
				const int word_index = word->index_;
				auto& l1 = syn0_[current_word->index_];

				// udate parameters by Hierarchical Softmax
				std::fill(grad_syn0.begin(), grad_syn0.end(), 0);
				syn1_train_->train_syn1(word, l1, alpha, grad_syn0); // update syn1_
				v::saxpy(l1, 1.0, grad_syn0); // syn0_[word_index] += grad_syn0;
			} // end loop: window
			++count;
		} // end loop: tokens in corpus

		return count;
	}

	float similarity(const String& w1, const String& w2) const {
		auto it1 = vocab_.find(w1), it2 = vocab_.find(w2);
		if (it1 != vocab_.end() && it2 != vocab_.end())
			return v::dot(syn0_[it1->second->index_], syn0_[it2->second->index_]);
		return 0;
	}

	// training strategy
	std::string syn1_train_method_; // "NS" : negative sampling, "HS" : hierarchical softmax
	std::shared_ptr<Syn1TrainStrategy<Word>> syn1_train_;
};

