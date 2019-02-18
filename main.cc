// OpenMP is required..
// g++-4.8 -ow2v -fopenmp -std=c++0x -Ofast -march=native -funroll-loops  main.cc  -lpthread

#include "word2vec.h"
#include <iostream>
#include <string>
#include <initializer_list>
#include <boost/program_options.hpp>

int accuracy(Word2Vec<std::string>& model, std::string questions, int restrict_vocab = 30000) {
	std::ifstream in(questions);
	std::string line;
	auto lower = [](std::string& data) { std::transform(data.begin(), data.end(), data.begin(), ::tolower);};
	size_t count = 0, correct = 0, ignore = 0, almost_correct = 0;
	const int topn = 10;
	while (std::getline(in, line)) {
		if (line[0] == ':') {
			printf("%s\n", line.c_str());
			continue;
		}

		std::istringstream iss(line);
		std::string a, b, c, expected;
		iss >> a >> b >> c >> expected;
		lower(a); lower(b); lower(c); lower(expected);

		if (!model.has(a) || !model.has(b) || !model.has(c) || !model.has(expected)) {
			printf("unhandled: %s %s %s %s\n", a.c_str(), b.c_str(), c.c_str(), expected.c_str());
			++ignore;
			continue;
		}

		++count;
		std::vector<std::string> positive{b, c}, negative{a};
		auto predict = model.most_similar(positive, negative, topn);
		if (predict[0].first == expected) { ++ correct; ++almost_correct; }
		else {
			bool found = false;
			for (auto& v: predict) {
				if (v.first == expected) { found = true; break; }
			}
			if (found) ++almost_correct;
			else printf("predicted: %s, expected: %s\n", predict[0].first.c_str(), expected.c_str());
		}
	}

	if (count > 0) printf("predict %lu out of %lu (%f%%), almost correct %lu (%f%%) ignore %lu\n", correct, count, correct * 100.0 / count, almost_correct, almost_correct * 100.0 / count, ignore);

	return 0;
}

int main(int argc, const char *argv[])
{
	// parse options
	namespace po = boost::program_options;
	po::options_description description("Allowed options for word2vec");
	description.add_options()
		("help,h", "help.")
		("mode,m", po::value<std::string>()->default_value("train"), "Mode train/test.")
		("output,o", po::value<std::string>()->default_value("./vectors.bin"), "Output path.")
		("dim,d", po::value<int>()->default_value(300), "Dimensionality of word embedding.")
		("window,w", po::value<int>()->default_value(5), "Window size.")
		("sample,s", po::value<float>()->default_value(0.001), "Subsampling probability.")
		("min-count,c", po::value<int>()->default_value(5), "The minimum frequency of words.")
		("negative,n", po::value<int>()->default_value(5), "The number of negative samples.")
		("alpha,a", po::value<float>()->default_value(0.025), "The initial learning rate.")
		("min-alpha,b", po::value<float>()->default_value(0.0001), "The minimum learning rate.")
		("n_workers,p", po::value<int>()->default_value(0), "The number of threads")
		("format,f", po::value<std::string>()->default_value("bin"), "Output file format: bin/text");

	po::positional_options_description pos_description;
	pos_description.add("input", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(description).positional(pos_description).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << "Usage : " << argv[0] << " [options] input_path" << std::endl;
		std::cout << description << std::endl;
		return 0;
	}

	const auto input_path = vm["input"].as<std::string>();
	const auto output_path = vm["output"].as<std::string>();
	const auto mode = vm["mode"].as<std::string>();
	const auto dim = vm["dim"].as<int>();
	const auto window = vm["window"].as<int>();
	const auto sample = vm["sample"].as<float>();
	const auto min_count = vm["min-count"].as<int>();
	const auto negative = vm["negative"].as<int>();
	const auto alpha = vm["alpha"].as<float>();
	const auto min_alpha = vm["min-alpha"].as<float>();
	const auto n_workers = vm["n_workers"].as<int>();
	const auto file_format = vm["format"].as<std::string>();

	// simple check for options
	if ( mode != "train" && mode != "test")
		throw po::validation_error(po::validation_error::invalid_option_value, "mode");

	if ( file_format != "bin" && file_format != "text")
		throw po::validation_error(po::validation_error::invalid_option_value, "format");

	// initalize model
	Word2Vec<std::string> model(dim, window, sample, min_count, negative, alpha, min_alpha);
	using Sentence = Word2Vec<std::string>::Sentence;
	using SentenceP = Word2Vec<std::string>::SentenceP;

	// model.phrase_ = true;

	::srand(::time(NULL));

	auto distance = [&model]() {
		while (1) {
			std::string s;
			std::cout << "\nFind nearest word for (:quit to break):";
			std::cin >> s;
			if (s == ":quit") break;
			auto p = model.most_similar(std::vector<std::string>{s}, std::vector<std::string>(), 10);
			size_t i = 0;
			for (auto& v: p) {
				std::cout << i++ << " " << v.first << " " << v.second << std::endl;
			}
		}
	};

	using time_point_t = decltype(std::chrono::high_resolution_clock::now());
	auto time_diff_sec = [](const time_point_t& start, const time_point_t& end) -> double {
		return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
	};

	if (mode == "train") {
		std::vector<SentenceP> sentences;

		size_t count =0;
		const size_t max_sentence_len = 200;

		SentenceP sentence(new Sentence);
		std::ifstream in(input_path);
		while (true) {
			std::string s;
			in >> s;
			if (s.empty()) break;

			++count;
			sentence->tokens_.push_back(std::move(s));
			if (count == max_sentence_len) {
				count = 0;
				sentence->words_.reserve(sentence->tokens_.size());
				sentences.push_back(std::move(sentence));
				sentence.reset(new Sentence);
			}
		}

		if (!sentence->tokens_.empty())
			sentences.push_back(std::move(sentence));

		auto cstart = std::chrono::high_resolution_clock::now();
		model.build_vocab(sentences);
		auto cend = std::chrono::high_resolution_clock::now();
		printf("load vocab: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		cstart = cend;
		model.train(sentences, n_workers);
		cend = std::chrono::high_resolution_clock::now();
		printf("train: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		cstart = cend;
		if (file_format == "bin") {
			model.save(output_path);
		}
		else {
			model.save_text(output_path);
		}
		cend = std::chrono::high_resolution_clock::now();
		printf("save model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);
	}

	if (mode == "test") {
		auto cstart = std::chrono::high_resolution_clock::now();
		if (file_format == "bin") {
			model.load(input_path);
		}
		else {
			model.load_text(input_path);
		}
		auto cend = std::chrono::high_resolution_clock::now();
		printf("load model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);

		distance();

		cstart = cend;
		accuracy(model, "questions-words.txt");
		cend = std::chrono::high_resolution_clock::now();
		printf("test model: %.4f seconds\n", std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count() / 1000000.0);
	}

	return 0;
}

