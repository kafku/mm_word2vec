#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <list>
#include <unordered_map>

//struct Code {
//	std::vector<uint32_t> points_; // index of nodes on the path from leaf to root
//	std::vector<uint8_t> codes_; // code of each nodes in huffman tree
//};

template <typename Index> struct _Node;
template <typename Index> using _NodeP = std::shared_ptr<_Node<Index>>;

template <typename Index>
struct _Node
{
	using PointList = std::vector<Index>;
	using CodeList = std::vector<uint8_t>;

	Index index_; // node index
	unsigned int count_;
	_NodeP<Index> left_, right_;

	_Node(const Index index, unsigned int count, _NodeP<Index> left = nullptr, _NodeP<Index> right = nullptr)
		: index_(index), count_(count), left_(left), right_(right) {}

	_Node(const _Node<Index>&) = delete;
	const _Node<Index>& operator=(const _Node<Index>&) = delete;

	bool is_leaf() const {
		return left_ == nullptr && right_ == nullptr;
	}

	PointList points_; // index of nodes on the path from leaf to root
	CodeList codes_; // code of each nodes in huffman tree
};


template <typename T> //NOTE: T must have index_ and counts_ as its member variable
class HuffmanTree
{
public:
	using Index = typename T::Index;
	using Node = _Node<Index>;
	using NodeP = _NodeP<Index>;

	HuffmanTree() = default;
	~HuffmanTree() = default;

	void build_tree(const std::vector<T*>& words);
	inline const typename Node::CodeList& encode(const T* const words) {
		return node_map[words->index_]->codes_;
	};
	inline const typename Node::PointList& get_points(const T* const words) {
		return node_map[words->index_]->points_;
	}

private:
	std::unordered_map<typename T::Index, NodeP> node_map;
};


template <typename T>
void HuffmanTree<T>::build_tree(const std::vector<T*>& words) {
	const auto n_words = words.size();

	auto comp = [](const NodeP n1, const NodeP n2) { return n1->count_ > n2->count_; };
	std::vector<NodeP> leaf_nodes(n_words);
	for (const auto& word : words) {
		leaf_nodes.emplace_back(std::make_shared<Node>(word->index_, word->count_)); // add leaf nodes == words
	}
	std::vector<NodeP> heap(leaf_nodes.size());
	std::copy(leaf_nodes.begin(), leaf_nodes.end(), std::back_inserter(heap));
	std::make_heap(heap.begin(), heap.end(), comp);

	// create the intermediate nodes
	for (int i = 0; i < n_words - 1; ++i) {
		std::pop_heap(heap.begin(), heap.end(), comp);
		auto min1 = heap.back(); heap.pop_back();
		std::pop_heap(heap.begin(), heap.end(), comp);
		auto min2 = heap.back(); heap.pop_back();

		heap.push_back(std::make_shared<Node>(i, min1->count_ + min2->count_, min1, min2)); // add an intermediate node
		std::push_heap(heap.begin(), heap.end(), comp);
	}

	// set codes and IDs to leaf nodes
	int max_depth = 0;
	std::list<std::tuple<NodeP, std::vector<Index>, std::vector<uint8_t>>> child_node_stack;
	child_node_stack.push_back(std::make_tuple(heap[0], std::vector<Index>(), std::vector<uint8_t>()));
	while (!child_node_stack.empty()) {
		auto t = child_node_stack.back();
		child_node_stack.pop_back();

		NodeP node = std::get<0>(t);
		if (node->is_leaf()) {
			node->points_ = std::get<1>(t);
			node->codes_ = std::get<2>(t);
			max_depth = std::max((int)node->codes_.size(), max_depth);
		}
		else {
			auto points = std::get<1>(t);
			points.emplace_back(node->index_);
			auto codes1 = std::get<2>(t);
			auto codes2 = codes1;
			codes1.push_back(0);
			codes2.push_back(1);
			child_node_stack.emplace_back(std::make_tuple(node->left_, points, codes1));
			child_node_stack.emplace_back(std::make_tuple(node->right_, points, codes2));
		}
	}

	std::cout << "built huffman tree with maximum node depth " << max_depth << std::endl;

	// create map (index ----> nodes)
	for (auto& node : leaf_nodes) {
		node_map[node->index_] = node;
	}
}
