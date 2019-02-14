#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <stack>
#include <unordered_map>


template <typename T> //NOTE: T must have idex_ and counts_ as its member variable
class HuffmanTree
{
public:
	struct Node;
	using NodeP = std::shared_ptr<Node>;
	struct Node
	{
		int32_t index_; // node index FIXME: what if T::index_ is not int32_t?
		unsigned int count_;
		NodeP left_, right_;

		Node(const int32_t index, unsigned int count, NodeP left = nullptr, NodeP right = nullptr)
			: index_(index), count_(count), left_(left), right_(right) {}

		Node(const Node&) = delete;
		const Node& operator=(const Node&) = delete;

		std::vector<uint32_t> points_; // index of nodes on the path from leaf to root
		std::vector<uint8_t> codes_; // code of each nodes in huffman tree
	};

	struct Leaf : public Node
	{
		Leaf(const T* const pt, const int32_t index, unsigned int count)
			: word_pt(pt), Node(index, count) {}

		const T * const word_pt;
	};

	using LeafP = std::shared_ptr<Leaf>;

	void build_tree(const std::vector<T*>& words);
	// TODO: how to get codes and points?

private:
	std::vector<LeafP> leaf_nodes;
};


template <typename T>
void HuffmanTree<T>::build_tree(const std::vector<T*>& words) {
	const auto n_words = words.size();
	//syn1_.resize(n_words); //FIXME: n_words -> # of nodes

	auto comp = [](const Node *n1, const Node *n2) { return n1->count_ > n2->count_; };
	leaf_nodes.resize(n_words);
	auto max_word_id = words[0]->index_;
	for (const auto& word : words) {
		leaf_nodes.emplace_back(std::make_shared(word->index_, word->count_)); // add leaf nodes == words
		max_word_id = std::max(max_word_id, word->index_);
	}
	std::vector<NodeP> heap(leaf_nodes.size());
	std::copy(leaf_nodes.begin(), leaf_nodes.end(), std::back_inserter(heap));
	std::make_heap(heap.begin(), heap.end(), comp);

	// create the intermediate nodes
	//std::vector<NodeP> intermediate_nodes(leaf_nodes.size());
	for (int i = 0; i < n_words - 1; ++i) {
		std::pop_heap(heap.begin(), heap.end(), comp);
		auto min1 = heap.back(); heap.pop_back();
		std::pop_heap(heap.begin(), heap.end(), comp);
		auto min2 = heap.back(); heap.pop_back();

		heap.push_back(std::make_shared(i + max_word_id, min1->count_ + min2->count_, min1, min2)); // add intermediate nodes, whose IDs > max_word_id
		std::push_heap(heap.begin(), heap.end(), comp);
	}

	// set codes and IDs to
	int max_depth = 0;
	std::stack<std::tuple<NodeP, std::vector<uint32_t>, std::vector<uint8_t>>> child_node_stack;
	child_node_stack.push_back(std::make_tuple(heap[0], std::vector<uint32_t>(), std::vector<uint8_t>()));
	while (!child_node_stack.empty()) {
		auto t = child_node_stack.back();
		child_node_stack.pop_back();

		NodeP node = std::get<0>(t);
		if (node->index_ < n_words) { // if the node is not an intermediate node i.e. leaf
			node->points_ = std::get<1>(t);
			node->codes_ = std::get<2>(t);
			max_depth = std::max((int)node->codes_.size(), max_depth);
		}
		else {
			auto points = std::get<1>(t);
			points.emplace_back(node->index_ - n_words);
			auto codes1 = std::get<2>(t);
			auto codes2 = codes1;
			codes1.push_back(0);
			codes2.push_back(1);
			child_node_stack.emplace_back(std::make_tuple(node->left_, points, codes1));
			child_node_stack.emplace_back(std::make_tuple(node->right_, points, codes2));
		}
	}

	std::cout << "built huffman tree with maximum node depth " << max_depth << std::endl;
}
