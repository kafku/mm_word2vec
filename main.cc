// OpenMP is required..
// g++-4.8 -ow2v -fopenmp -std=c++0x -Ofast -march=native -funroll-loops  main.cc  -lpthread

#include "word2vec.h"
#include "grad_utils.h"
#include <iostream>
#include <string>
#include <initializer_list>
#include <boost/program_options.hpp>
#include <glog/logging.h>

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
	// seg glog
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();

	// parse options
	namespace po = boost::program_options;
	po::options_description description("Allowed options");
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
		("n_workers,p", po::value<int>()->default_value(0), "The number of threads. Use all CPUs when 0.")
		("format,f", po::value<std::string>()->default_value("bin"), "Output file format: bin/text")
		("iteration,i", po::value<int>()->default_value(5), "The number of iterations")
		("method,M", po::value<std::string>()->default_value("HS"), "Methods: HierarchicalSoftmax(HS)/NegativeSampling(NS)")
		("multimodal-input,I", po::value<std::string>()->default_value(""), "Path to multimodal feature file")
		("mul-negative", po::value<int>()->default_value(5), "The number of negative images.")
		("mul-margin", po::value<float>()->default_value(0.5), "Margin used in the multimodal objective")
		("mul-regparam", po::value<float>()->default_value(0.0001), "Regularization parameter used in the multimodal objective")
		("mul-min-count", po::value<int>()->default_value(100), "The minimum frequency of words associated with images.")
		("mul-output", po::value<std::string>()->default_value("./lt_mat.hdf5"), "Output path of linear transformation.")
		("input_path", po::value<std::string>()->required(), "Path to input file");

	po::positional_options_description pos_description;
	pos_description.add("input_path", 1);

	po::variables_map vm;
	try {
		po::store(po::command_line_parser(argc, argv).options(description).positional(pos_description).run(), vm);
		po::notify(vm);
	}
	catch(std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cout << "Usage : " << argv[0] << " [options] input_path" << std::endl;
		std::cout << description << std::endl;
		return 1;
	}

	if (vm.count("help")) {
		std::cout << "Usage : " << argv[0] << " [options] input_path" << std::endl;
		std::cout << description << std::endl;
		return 0;
	}

	const auto input_path = vm["input_path"].as<std::string>();
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
	const auto n_iterations = vm["iteration"].as<int>();
	const auto train_method = vm["method"].as<std::string>();
	const auto multimodal_path = vm["multimodal-input"].as<std::string>();
	const auto mul_n_negative = vm["mul-negative"].as<int>();
	const auto mul_margin = vm["mul-margin"].as<float>();
	const auto mul_regparam = vm["mul-regparam"].as<float>();
	const auto mul_min_count = vm["mul-min-count"].as<int>();
	const auto mul_output_path = vm["mul-output"].as<std::string>();

	// simple check for options
	if ( mode != "train" && mode != "test")
		throw po::validation_error(po::validation_error::invalid_option_value, "mode");

	if ( file_format != "bin" && file_format != "text")
		throw po::validation_error(po::validation_error::invalid_option_value, "format");

	if ( train_method != "HS" && train_method != "NS")
		throw po::validation_error(po::validation_error::invalid_option_value, "method");

	std::cout << "Input file: " << input_path << std::endl;

	// initalize model
	Word2Vec<std::string> model(dim, window, sample, min_count, alpha, min_alpha, n_iterations);
	using Sentence = Word2Vec<std::string>::Sentence;
	using SentenceP = Word2Vec<std::string>::SentenceP;
	using Word = Word2Vec<std::string>::Word;

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

		// set training strategies
		const auto n_words = model.words_.size();
		const auto layer1_size = model.layer1_size_;
		if (train_method == "HS") { // Hierarchical Softmax
			std::cout << "Training method: Hierarchical Softmax" << std::endl;
			std::shared_ptr<HierarchicalSoftmax<Word>> HS_strategy = std::make_shared<HierarchicalSoftmax<Word>>(layer1_size, n_words);
			HS_strategy->build_tree(model.words_);
			model.syn1_train_ = HS_strategy;
		}
		else if (train_method == "NS") { // Negative Sampling
			std::cout << "Training method: Negative Sampling" << std::endl;
			std::cout << "  # of negative samples : " << negative << std::endl;
			std::shared_ptr<NegativeSampling<Word>> NS_strategy = std::make_shared<NegativeSampling<Word>>(layer1_size, n_words, negative);
			NS_strategy->update_distribution(model.words_);
			model.syn1_train_ = NS_strategy;
		}
		else {
			// NOTE: invalid option
			// TODO: show error message
			return -1;
		}

		if (multimodal_path.size() == 0) {
			model.syn0_train_ = std::make_shared<SimpleGD<Word>>(layer1_size, n_words);
		}
		else {
			// FIXME: fix hard-coded values
			auto MMGD_strategy = std::make_shared<MultimodalGD<Word, gu::CosSim<float>>>(
					layer1_size, n_words, mul_n_negative, mul_margin, mul_regparam, mul_min_count);
			MMGD_strategy->save_lt_on_exit(mul_output_path);
			MMGD_strategy->load(multimodal_path, true);
			model.syn0_train_ = MMGD_strategy;
		}

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

