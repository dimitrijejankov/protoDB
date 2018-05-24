#pragma once

#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <assert.h>
#include <thread>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <memory>
#include <atomic>
#include "immintrin.h"
// #include "smmintrin.h"
#include "cycletimer.h"
#include "barrier.h"
#include <jemalloc/jemalloc.h>

using std::cout;
using std::endl;

#define UNUSED __attribute__((unused))

namespace palmtree {

  static std::atomic<int> NODE_NUM(0);
  unsigned int batch_id = 0;
  /**
   * Tree operation types
   */
  enum TreeOpType {
    TREE_OP_FIND = 0,
    TREE_OP_INSERT,
    TREE_OP_REMOVE
  };

  enum NodeType {
    INNERNODE = 0,
    LEAFNODE
  };

  class Stats {
  public:
    Stats(int worker_num): worker_num_(worker_num) {}
    Stats() {}
    /**
     * add stat for one metric of one worker
     */
    void add_stat(int worker_id, std::string metric_name, double metric_value) {
      stats_[metric_name][worker_id] += metric_value;
    }

    void init_metric(std::string metric_name) {
      stats_[metric_name] = std::vector<cycletimer::SysClock>(worker_num_);
      for (int i = 0; i < worker_num_; i++)
        stats_[metric_name][i] = 0;
      metric_names_.push_back(metric_name);
    }

    /**
     * Print the stats out
     */
    void print_stat() {
      for (auto &metric_name : metric_names_) {
        auto &values = stats_[metric_name];
        std::string line = "";
        for (int i = 0; i < worker_num_; i++) {
          if (metric_name == "leaf_task") {
            line += "\t" + std::to_string(i) + ": " + std::to_string(values[i]);

          } else {
            line += "\t" + std::to_string(i) + ": " + std::to_string(values[i] * cycletimer::secondsPerTick());
          }
        }
      }
    }

    void reset_metric() {
      for (auto itr = stats_.begin(); itr != stats_.end(); itr++) {
        for (int i = 0; i < worker_num_; i++) {
          itr->second[i] = 0;
        }
      }
    }
  private:
    std::unordered_map<std::string, std::vector<cycletimer::SysClock>> stats_;
    std::vector<std::string> metric_names_;
    int worker_num_;
  } STAT;

  template <typename KeyType,
           typename ValueType,
           typename PairType = std::pair<KeyType, ValueType>,
           typename KeyComparator = std::less<KeyType> >
  class PalmTree {
  public:
    // Number of working threads
    int NUM_WORKER;
    int BATCH_SIZE;

  private:
    // Max number of slots per inner node
    static const int INNER_MAX_SLOT = 256;
    // Max number of slots per leaf node
    static const int LEAF_MAX_SLOT = 64;
    // Threshold to control bsearch or linear search
    static const int BIN_SEARCH_THRESHOLD = 32;
    // Number of working threads
    static const int BATCH_SIZE_PER_WORKER = 4096;

  private:
    /**
     * Tree node base class
     */
    struct InnerNode;
    struct Node {
      // Number of actually used slots
      int slot_used;
      int id;
      int level;
      KeyType lower_bound;
      Node *parent;


      Node() = delete;
      Node(Node *p, int lvl): slot_used(0), level(lvl), parent(p) {
        id = NODE_NUM++;
      };
      virtual ~Node() {};
      virtual std::string to_string() = 0;
      virtual NodeType type() const = 0;
      virtual bool is_few() = 0;
    };

    struct InnerNode : public Node {
      InnerNode() = delete;
      InnerNode(Node *parent, int level): Node(parent, level){};
      virtual ~InnerNode() {};
      // Keys for values
      KeyType keys[LEAF_MAX_SLOT];
      // Pointers for child nodes
      Node *values[LEAF_MAX_SLOT];

      virtual NodeType type() const {
        return INNERNODE;
      }

      virtual std::string to_string() {
        std::string res;
        res += "InnerNode[" + std::to_string(Node::id) + " @ " + std::to_string(Node::level) + "] ";
        for (int i = 0 ; i < Node::slot_used ; i++) {
          res += " " + std::to_string(keys[i]) + ":" + std::to_string(values[i]->id);
        }
        return res;
      }

      inline bool is_full() const {
        return Node::slot_used == MAX_SLOT();
      }


      inline size_t MAX_SLOT() const {
        return LEAF_MAX_SLOT;
      }

      virtual inline bool is_few() {
        return Node::slot_used < MAX_SLOT()/4 || Node::slot_used == 0;
      }

    };

    struct LeafNode : public Node {
      LeafNode() = delete;
      LeafNode(Node *parent, int level): Node(parent, level){};
      virtual ~LeafNode() {};

      // Keys and values for leaf node
      KeyType keys[INNER_MAX_SLOT];
      ValueType values[INNER_MAX_SLOT];

      virtual NodeType type() const {
        return LEAFNODE;
      }

      virtual std::string to_string() {
        std::string res;
        res += "LeafNode[" + std::to_string(Node::id) + " @ " + std::to_string(Node::level) + "] ";

        for (int i = 0 ; i < Node::slot_used ; i++) {
          res += " " + std::to_string(keys[i]) + ":" + std::to_string(values[i]);
        }
        return res;
      }

      inline bool is_full() const {
        return Node::slot_used == MAX_SLOT();
      }

      inline size_t MAX_SLOT() const {
        return INNER_MAX_SLOT;
      }

      virtual inline bool is_few() {
        return Node::slot_used < MAX_SLOT()/4 || Node::slot_used == 0;
      }
    };
    /**
     * Tree operation wrappers
     */
    struct TreeOp {
      // Op can either be none, add or delete
      TreeOp(TreeOpType op_type, const KeyType &key, const ValueType &value):
        op_type_(op_type), key_(key), value_(value), target_node_(nullptr),
        boolean_result_(false), done_(false) {};


      TreeOp(TreeOpType op_type, const KeyType &key):
        op_type_(op_type), key_(key), target_node_(nullptr),
        boolean_result_(false), done_(false) {};

      TreeOpType op_type_;
      KeyType key_;
      ValueType value_;

      LeafNode *target_node_;
      ValueType result_;
      bool boolean_result_;
      bool done_;

      // Wait until this operation is done
      // Now use busy waiting, should use something more smart. But be careful
      // that conditional variable could be very expensive
      inline void wait() {
        while (!done_) {
          boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
        }
      }
    };

    /**
     * A batch of tree operations, this data structure is not thread safe
     * The major goal of this class is to amortize memory allocation of
     * tree operations
     */
    class TaskBatch {
    public:
      TaskBatch(size_t capacity): capacity_(capacity), ntask_(0) {
        ops_ = (TreeOp *)malloc(sizeof(TreeOp) * capacity_);
      }

      void destroy() {
        free(ops_);
        ops_ = nullptr;
      }

      // Add a tree operation to the batch
      inline void add_op(TreeOpType op_type, const KeyType *keyp, const ValueType *valp) {
        assert(ntask_ != capacity_);

        if (op_type == TREE_OP_INSERT) {
          assert(valp != nullptr);
          ops_[ntask_++] = TreeOp(op_type, *keyp, *valp);
        } else {
          ops_[ntask_++] = TreeOp(op_type, *keyp);
        }
      }

      // Whether the tree is full or not
      inline bool is_full() { return ntask_ == capacity_; }
      // The size of the batch
      inline size_t size() { return ntask_; }
      // Overloading [] to return the ith operation in the batch
      TreeOp * get_op(int i) {
        assert(i < ntask_);
        return ops_ + i;
      }

      // Capacity of the batch
      size_t capacity_;
      // Number of tasks currently in the batch
      size_t ntask_;
      // Tree opearations
      TreeOp *ops_;
    };

    enum ModType {
      MOD_TYPE_ADD,
      MOD_TYPE_DEC,
      MOD_TYPE_NONE
    };

    /**
     * Wrapper for node modification
     */
    struct NodeMod {
      NodeMod(ModType type): type_(type) {}
      NodeMod(const TreeOp &op) {
        if (op.op_type_ == TREE_OP_REMOVE) {
          this->type_ = MOD_TYPE_DEC;
          this->value_items.emplace_back(std::make_pair(op.key_, ValueType()));
        } else {
          this->type_ = MOD_TYPE_ADD;
          this->value_items.emplace_back(std::make_pair(op.key_, op.value_));
        }
      }
      ModType type_;
      // For leaf modification
      std::vector<std::pair<KeyType, ValueType>> value_items;
      // For inner node modification
      std::vector<std::pair<KeyType, Node *>> node_items;
      // For removed keys
      std::vector<std::pair<KeyType, ValueType>> orphaned_kv;
    };

  /********************
   * PalmTree private
   * ******************/
  private:
    // Root of the palm tree
    Node *tree_root;
    // Height of the tree
    int tree_depth_;
    // Number of nodes on each layer
    std::vector<std::atomic<int> *> layer_width_;
    // Is the tree being destroyed or not
    bool destroyed_;
    // Minimal key
    KeyType min_key_;
    // Key comparator
    KeyComparator kcmp;
    // Current batch of the tree
    TaskBatch *tree_current_batch_;

    // Push a task into the current batch, if the batch is full, push the batch
    // into the batch queue.
    void push_task(TreeOpType op_type, const KeyType *keyp, const ValueType *valp) {
      tree_current_batch_->add_op(op_type, keyp, valp);
      task_nums += 2;

      if (tree_current_batch_->is_full()) {
        task_batch_queue_.push(tree_current_batch_);
        tree_current_batch_ = (TaskBatch *)malloc(sizeof(TaskBatch));
        new (tree_current_batch_) TaskBatch(BATCH_SIZE);
      }
    }

    // Return true if k1 < k2
    inline bool key_less(const KeyType &k1, const KeyType &k2) {
      return kcmp(k1, k2);
    }
    // Return true if k1 == k2
    inline bool key_eq(const KeyType &k1, const KeyType &k2) {
      return !kcmp(k1, k2) && !kcmp(k2, k1);
    }


    // Return the index of the largest slot whose key <= @target
    // assume there is no duplicated element
    int search_helper(const KeyType *input, int size, const KeyType &target) {
      int res = -1;
      // loop all element
      for (int i = 0; i < size; i++) {
        if(key_less(target, input[i])){
          // target < input
          // ignore
          continue;

        }
        if (res == -1 || key_less(input[res], input[i])) {
          res = i;
        }
      }

      return res;
    }

    int search_leaf(const KeyType *data, int size, const KeyType &target) {
      const __m256i keys = _mm256_set1_epi32(target);

      const auto n = size;
      const auto rounded = 8 * (n/8);

      for (int i=0; i < rounded; i += 8) {

        const __m256i vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data[i]));

        const __m256i cmp1 = _mm256_cmpeq_epi32(vec1, keys);

        const uint32_t mask = _mm256_movemask_epi8(cmp1);

        if (mask != 0) {
          return i + __builtin_ctz(mask)/4;
        }
      }

      for (int i = rounded; i < n; i++) {
        if (data[i] == target) {
          return i;
        }
      }

      return -1;
    }



    // Return the index of the largest slot whose key <= @target
    // assume there is no duplicated element
    int search_inner(const KeyType *input, int size, const KeyType &target) {

      int low = 0, high = size - 1;
      while (low != high) {
        int mid = (low + high) / 2 + 1;
        if (key_less(target, input[mid])) {
          high = mid - 1;
        }
        else {
          low = mid;
        }
      }

      if (low == size) {
        return -1;
      }
      return low;
    }

    /**
     * @brief Return the leaf node that contains the @key
     */
    LeafNode *search(const KeyType &key UNUSED) {

      auto ptr = (InnerNode *)tree_root;
      for (;;) {
        auto idx = this->search_inner(ptr->keys, ptr->slot_used, key);
        Node *child = ptr->values[idx];
        if (child->type() == LEAFNODE) {
          return (LeafNode *)child;
        } else {
          ptr = (InnerNode *)child;
        }
      }
      // we shouldn't reach here
      assert(0);
    }

    /**
     * @brief big_split will split the kv pair vector into multiple tree nodes
     *  that is within the threshold. The actual type of value is templated as V.
     *  The splited nodes should be stored in Node, respect to appropriate
     *  node types
     */
    template<typename NodeType, typename V>
    void big_split(std::vector<std::pair<KeyType, V>> &input, NodeType *node, std::vector<std::pair<KeyType, Node *>> &new_nodes) {
      std::sort(input.begin(), input.end(), [this](const std::pair<KeyType, V> &p1, const std::pair<KeyType, V> &p2) {
        return key_less(p1.first, p2.first);
      });

      auto itr = input.begin();

      auto item_per_node = node->MAX_SLOT() / 2;
      auto node_num = input.size() / (item_per_node);
      // save first half items (small part) in old node
      node->slot_used = 0;
      for (int i = 0; i < item_per_node; i++) {
        // add_item<NodeType, V>(node, itr->first, itr->second);
        node->keys[i] = itr->first;
        node->values[i] = itr->second;
        node->slot_used++;
        itr++;
      }

      // Add a new node
      int node_create_num = 1;
      while(node_create_num < node_num) {

        NodeType *new_node = new NodeType(node->parent, node->Node::level);
        layer_width_[node->Node::level]->fetch_add(1);

        // save the second-half in new node
        auto new_key = (*itr).first;
        int i = 0;
        while (itr != input.end() && new_node->slot_used < item_per_node) {
          // add_item<NodeType, V>(new_node, itr->first, itr->second);
          new_node->keys[i] = itr->first;
          new_node->values[i] = itr->second;
          new_node->slot_used++;
          itr++;
          i++;
        }
        if(node_create_num == node_num - 1) {
          while(itr != input.end()) {
            new_node->keys[i] = itr->first;
            new_node->values[i] = itr->second;
            new_node->slot_used++;
            itr++;
            i++;
          }
        }

        new_nodes.push_back(std::make_pair(new_key, new_node));
        node_create_num++;
      }
    }

    // Warning: if this function return true, the width of the layer will be
    // decreased by 1, so the caller must actually merge the node
    bool must_merge(Node *node) {
      if (!node->is_few())
        return false;

      int old_width = layer_width_[node->level]->fetch_add(-1);
      if (old_width == 1) {
        // Can't merge
        layer_width_[node->level]->fetch_add(1);
        return false;
      }

      return true;
    }

    template <typename NodeType, typename V>
    void add_item(NodeType *node, const KeyType &key, V value) {
      // add item to leaf node
      // just append it to the end of the slot
      if (node->type() == LEAFNODE) {
        // auto idx = node->slot_used++;
        auto idx = search_leaf(node->keys, node->slot_used, key);
        if(idx != -1) {
          return;
        }
        idx = node->slot_used++;
        node->keys[idx] = key;
        node->values[idx] = value;
        return;
      }

      if(node->slot_used == 0) {
        node->keys[0] = key;
        node->values[0] = value;
        node->slot_used++;
        return;
      }

      // add item to inner node
      // ensure it's order
      auto idx = search_inner(node->keys, node->slot_used, key);
      auto k = key;
      auto v = value;

      for(int i = idx + 1; i < node->slot_used; i++) {
        std::swap(node->keys[i], k);
        std::swap(node->values[i], v);
      }

      node->keys[node->slot_used] = k;
      node->values[node->slot_used] = v;
      node->slot_used++;
    }


    template <typename NodeType>
    void del_item(NodeType *node, const KeyType &key) {
      auto lastIdx = node->slot_used - 1;
      auto idx = search_helper(node->keys, node->slot_used, key);
      if (idx == -1) {
        return;
      }

      if (!key_eq(key, node->keys[idx])) {
        if (node->type() == LEAFNODE)
          return;
      }

      if (node->type() == INNERNODE) {
        Node *child_node = reinterpret_cast<Node *>(&node->values[idx]);
        free_recursive(child_node);

        KeyType del_key = node->keys[idx];

        // auto k = node->keys[idx];
        // auto v = node->value[idx];
        for(int i = idx; i < node->slot_used - 1; i++) {
          std::swap(node->keys[i], node->keys[i + 1]);
          std::swap(node->values[i], node->values[i + 1]);
        }

        if(idx == 0) {
          node->keys[0] = del_key;
        }

        node->slot_used--;


      }else {
        // del in leaf
        if (idx == lastIdx) {
          // if it's the last element, just pop it
          node->slot_used--;
        } else {
          // otherwise, swap
          node->keys[idx] = node->keys[lastIdx];
          node->values[idx] = node->values[lastIdx];
          node->slot_used--;
        }
      }

      return;
    }

    // collect kv pairs in (or under) this node
    // used for merge
    void collect_leaf(Node *node, std::vector<std::pair<KeyType, ValueType>> &container) {
      if (node->type() == LEAFNODE) {
        auto ptr = (LeafNode *)node;
        for(int i = 0; i < node->slot_used; i++) {
          container.push_back(std::make_pair(ptr->keys[i], ptr->values[i]));
        }
      } else if (node->type() == INNERNODE) {
        auto ptr = (InnerNode *)node;
        for(int i = 0; i < node->slot_used; i++) {
          collect_leaf(ptr->values[i], container);
        }
        layer_width_[node->level-1]->fetch_add(-node->slot_used);
      } else {
        assert(0);
      }

      return;
    }

    /**
     * @brief Modify @node by applying node modifications in @modes. If @node
     * is a leaf node, @mods will be a list of add kv and del kv. If @node is
     * a inner node, @mods will be a list of add range and del range. If new
     * node modifications are triggered, record them in @new_mods.
     */
    NodeMod modify_node(Node *node, const std::vector<NodeMod> &mods) {
      if(node->type() == LEAFNODE) {
        return modify_node_leaf((LeafNode *)node, mods);
      } else{
        return modify_node_inner((InnerNode *)node, mods);
      }
    }

    NodeMod modify_node_leaf(LeafNode *node, const std::vector<NodeMod> &mods) {
      NodeMod ret(MOD_TYPE_NONE);
      auto& kv = ret.orphaned_kv;

      // randomly pick up a key, used for merge
      auto node_key = node->keys[0];

      // firstly, we loop all items to save orphaned and count nodes
      int num = node->slot_used;
      for (auto& item : mods) {
        // save all orphaned_*
        kv.insert(kv.end(), item.orphaned_kv.begin(), item.orphaned_kv.end());

        auto item_size = (int)item.value_items.size();
        if (item.type_ == MOD_TYPE_ADD) {
          num += item_size;
        } else if (item.type_ == MOD_TYPE_DEC) {
          num -= item_size;
        } else {
          assert(item_size == 0);
        }
      }

      if (num > node->MAX_SLOT()) {
        auto comp = [this](const std::pair<KeyType, ValueType> &p1, const std::pair<KeyType, ValueType> &p2) {
          return key_less(p1.first, p2.first);
        };

        std::set<std::pair<KeyType, ValueType>, decltype(comp)> buf(comp);

        // execute add/del
        for (auto& item : mods) {
          if (item.type_ == MOD_TYPE_ADD) {
            for (auto& kv : item.value_items) {
              buf.insert(kv);
            }
          } else if(item.type_ == MOD_TYPE_DEC) {
            for (auto& kv : item.value_items) {
              if(buf.count(kv)) {
                buf.erase(kv);
              }else{
                del_item<LeafNode>(node, kv.first);
              }
            }
          }
        }

        // construct input for split
        std::vector<std::pair<KeyType, ValueType>> split_input;
        for(auto itr = buf.begin(); itr != buf.end(); itr++) {
          split_input.push_back(*itr);
        }

        for(auto i = 0; i < node->slot_used; i++) {
          split_input.push_back(std::make_pair(node->keys[i], node->values[i]));
        }
        // do split based on this buf
        big_split<LeafNode, ValueType>(split_input, node, ret.node_items);
        ret.type_ = MOD_TYPE_ADD;
        return ret;
      } else {
        for (auto& item : mods) {
          if (item.type_ == MOD_TYPE_ADD) {
            for (auto& kv : item.value_items) {
              add_item<LeafNode, ValueType>(node, kv.first, kv.second);
            }
          } else if(item.type_ == MOD_TYPE_DEC) {
            for (auto& kv : item.value_items) {
              del_item<LeafNode>(node, kv.first);
            }
          }
        }
      }

      // merge
      // fixme: never merge the first leafnode
      // because the min_key is in this node
      // we can't delete min_key
      if (must_merge(node)) {
        collect_leaf(node, ret.orphaned_kv);
        ret.node_items.push_back(std::make_pair(node_key, node));
        ret.type_ = MOD_TYPE_DEC;
      }

      return ret;
    }

    NodeMod modify_node_inner(InnerNode *node UNUSED, const std::vector<NodeMod> &mods UNUSED) {
      NodeMod ret(MOD_TYPE_NONE);
      auto& kv = ret.orphaned_kv;

      // randomly pick up a key, used for merge
      auto node_key = node->keys[0];

      // firstly, we loop all items to save orphaned and count nodes
      int num = node->slot_used;
      for (auto& item : mods) {
        // save all orphaned_*
        kv.insert(kv.end(), item.orphaned_kv.begin(), item.orphaned_kv.end());

        auto item_size = (int)item.node_items.size();
        if (item.type_ == MOD_TYPE_ADD) {
          num += item_size;
        } else if (item.type_ == MOD_TYPE_DEC) {
          num -= item_size;
        } else {
          assert(item_size == 0);
        }
      }

      if (num > node->MAX_SLOT()) {
        auto comp = [this](const std::pair<KeyType, Node *> &p1, const std::pair<KeyType, Node *> &p2) {
          return key_less(p1.first, p2.first);
        };

        std::set<std::pair<KeyType, Node *>, decltype(comp)> buf(comp);

        // execute add/del
        for (auto& item : mods) {
          if (item.type_ == MOD_TYPE_ADD) {
            for (auto& kv : item.node_items) {
              buf.insert(kv);
            }
          } else if(item.type_ == MOD_TYPE_DEC) {
            for (auto& kv : item.node_items) {
              if(buf.count(kv)) {
                buf.erase(kv);
                // TODO: memleak
              }else{
                // cout << "del " << kv.first<<endl;
                del_item<InnerNode>(node, kv.first);

              }
            }
          }
        }

        // construct input for split
        std::vector<std::pair<KeyType, Node *>> split_input;
        for(auto itr = buf.begin(); itr != buf.end(); itr++) {
          split_input.push_back(*itr);
        }

        for(auto i = 0; i < node->slot_used; i++) {
          split_input.push_back(std::make_pair(node->keys[i], node->values[i]));
        }
        // do split based on this buf
        big_split<InnerNode, Node *>(split_input, node, ret.node_items);
        for (auto itr = ret.node_items.begin(); itr != ret.node_items.end(); itr++) {
          // Reset parent, the children of the newly splited node should point
          // to the new parent
          auto new_node = itr->second;
          for (int i = 0; i < new_node->slot_used; i++) {
            ((InnerNode *)new_node)->values[i]->parent = new_node;
          }
        }
        ret.type_ = MOD_TYPE_ADD;
        return ret;
      } else {
        for (auto& item : mods) {
          if (item.type_ == MOD_TYPE_ADD) {
            for (auto& kv : item.node_items) {
              add_item<InnerNode, Node *>(node, kv.first, kv.second);
            }
          } else if(item.type_ == MOD_TYPE_DEC) {
            for (auto& kv : item.node_items) {
              del_item<InnerNode>(node, kv.first);
            }
          }
        }
      }

      // merge
      if (must_merge(node)) {
        collect_leaf(node, ret.orphaned_kv);
        ret.node_items.push_back(std::make_pair(node_key, node));
        ret.type_ = MOD_TYPE_DEC;

      }

      return ret;
    }

    // set the smallest key in node to min_key
    void ensure_min_range(InnerNode *node UNUSED, const KeyType &min) {
      if (node->slot_used <= 1) {
        return;
      }
      // find the second smallest
      int idx = 0;
      for(int i = 1; i < node->slot_used; i++) {
        if(key_less(node->keys[i], node->keys[idx])) {
          idx = i;
        }
      }

      if(idx == 0) {
        return;
      }

      // swap idx with slot 0

      std::swap(node->keys[0], node->keys[idx]);
      std::swap(node->values[0], node->values[idx]);

    }

    void ensure_min_key() {
      auto ptr = (Node *)tree_root;
      while(ptr->type() == INNERNODE) {
        auto inner = (InnerNode *)ptr;
        inner->keys[0] = min_key_;
        ptr = inner->values[0];
      }
    }

    void ensure_tree_structure(Node *node, int indent) {
      std::map<int, int> recorder;
      ensure_tree_structure_helper(node, indent, recorder);
    }

    void ensure_tree_structure_helper(Node *node, int indent, std::map<int, int>& layer_size_recorder) {
      if(layer_size_recorder.count(node->level)) {
        layer_size_recorder[node->level]++;
      } else {
        layer_size_recorder[node->level] = 1;
      }
      std::string space;
      for (int i = 0; i < indent; i++)
        space += " ";

      if (node->type() == INNERNODE) {
        InnerNode *inode = (InnerNode *)node;
        for (int i = 0; i < inode->slot_used; i++) {
          auto child = inode->values[i];
        }
      }
      if (node->type() == INNERNODE) {
        InnerNode *inode = (InnerNode *)node;
        for (int i = 0; i < inode->slot_used; i++) {
          auto child = inode->values[i];
          KeyType *key_set;
          if (child->type() == LEAFNODE)
            key_set = ((LeafNode *)child)->keys;
          else
            key_set = ((InnerNode *)child)->keys;
          if (child->slot_used == 0) {
          } else {
            int idx = 0;
            for (int j = 1; j < child->slot_used; j++) {
              if (key_less(key_set[j], key_set[idx])) {
                idx = j;
              }
            }
          }
        }

        for (int i = 0; i < inode->slot_used; i++) {
          ensure_tree_structure_helper(inode->values[i], indent + 4, layer_size_recorder);
        }
      }
    }

    /**************************
     * Concurrent executions **
     *
     * Design: we have a potential infinite long task queue, where clients add
     * requests by calling find, insert or remove. We also have a fixed length
     * pool of worker threads. One of the thread (thread 0) will collect task from the
     * work queue, if it has collected enough task for a batch, or has timed out
     * before collecting enough tasks, it will partition the work and start the
     * Palm algorithm among the threads.
     * ************************/
    // boost::barrier barrier_;
    Barrier barrier_;
    boost::lockfree::spsc_queue<TaskBatch *> task_batch_queue_;

    // The current batch that is being processed by the workers
    TaskBatch *current_batch_;

    void sync(int worker_id) {
      auto begin_tick = cycletimer::currentTicks();
      barrier_.wait();
      auto passed_tick = cycletimer::currentTicks() - begin_tick;
      STAT.add_stat(worker_id, "sync_time", passed_tick);
    }

    struct WorkerThread {
      WorkerThread(int worker_id, PalmTree *palmtree):
        worker_id_(worker_id),
        palmtree_(palmtree),
        done_(false) {
          // Initialize 2 layers of modifications
          node_mods_.push_back(NodeModsMapType());
          node_mods_.push_back(NodeModsMapType());
        }
      // Worker id, the thread with worker id 0 will need to be the coordinator
      int worker_id_;
      // The work for the worker at each stage
      std::vector<TreeOp *> current_tasks_;
      std::unordered_map<Node *, std::vector<TreeOp *>> leaf_ops_;
      // Node modifications on each layer, the size of the vector will be the
      // same as the tree height
      typedef std::unordered_map<Node *, std::vector<NodeMod>> NodeModsMapType;
      std::vector<NodeModsMapType> node_mods_;
      // Spawn a thread and run the worker loop
      boost::thread wthread_;
      // The palm tree the worker belong to
      PalmTree *palmtree_;
      bool done_;
      void start() {
        wthread_ = boost::thread(&WorkerThread::worker_loop, this);
      }

      inline int LOWER() {
        auto batch_size = palmtree_->current_batch_->size();
        auto task_per_thread = batch_size / palmtree_->NUM_WORKER + 1;
        auto LOWER = worker_id_*task_per_thread;
        return LOWER;
      }

      inline int UPPER() {
        auto batch_size = palmtree_->current_batch_->size();
        auto task_per_thread = batch_size / palmtree_->NUM_WORKER + 1;
        auto LOWER = worker_id_*task_per_thread;
        return (worker_id_ == palmtree_->NUM_WORKER-1) ? (batch_size) : (LOWER+task_per_thread);
      }
      // The #0 thread is responsible to collect tasks to a batch
      void collect_batch() {

        if (worker_id_ == 0) {
          if (batch_id % 2 == 0) {
            int sleep_time = 0;
            while (sleep_time < 1024) {

              bool res = palmtree_->task_batch_queue_.pop(palmtree_->current_batch_);
              if (res) {
                break;
              } else {
                sleep_time++;
              }
            }
          }
          batch_id++;
          // STAT.add_stat(0, "fetch_batch", CycleTimer::currentTicks() - bt);
          // DLOG(INFO) << "Collected a batch of " << palmtree_->current_batch_->size();
        }

        palmtree_->sync(worker_id_);
        if (palmtree_->current_batch_ == nullptr) {
          return;
        }

        if (palmtree_->current_batch_->size() == 0) {
          return;
        }
        // STAT.add_stat(worker_id_, "batch_sort", CycleTimer::currentTicks() - bt);

        // Partition the task among threads
        int batch_size = palmtree_->current_batch_->size();
        int task_per_thread = batch_size / palmtree_->NUM_WORKER;
        int task_residue = batch_size - task_per_thread * palmtree_->NUM_WORKER;

        int lower = task_per_thread * worker_id_ + std::min(task_residue, worker_id_);
        int upper = lower + task_per_thread + (worker_id_ < task_residue);

        for (int i = lower; i < upper; i++) {
          palmtree_->workers_[worker_id_].current_tasks_
              .push_back(palmtree_->current_batch_->get_op(i));
        }
      }

      // Redistribute the tasks on leaf node
      void redistribute_leaf_tasks(std::unordered_map<Node *, std::vector<TreeOp *>> &result) {
        // First add current tasks
        for (auto op : current_tasks_) {
          if (result.find(op->target_node_) == result.end()) {
            result.emplace(op->target_node_, std::vector<TreeOp *>());
          }

          result[op->target_node_].push_back(op);
        }

        // Then remove nodes that don't belong to the current worker
        for (int i = 0; i < worker_id_; i++) {
          WorkerThread &wthread = palmtree_->workers_[i];
          for (int j = wthread.current_tasks_.size()-1; j >= 0; j--) {
            auto &op = wthread.current_tasks_[j];
            if (result.count(op->target_node_) == 0)
              break;
            result.erase(op->target_node_);
          }
        }

        for (int i = worker_id_+1; i < palmtree_->NUM_WORKER; i++) {
          WorkerThread &wthread = palmtree_->workers_[i];
          bool early_break = false;
          for (auto op : wthread.current_tasks_) {
            if (result.find(op->target_node_) != result.end()) {
              result[op->target_node_].push_back(op);
            } else {
              early_break = true;
              break;
            }
          }

          if (early_break)
            break;
        }

        // Calculate number of tasks
        int sum = 0;
        for (auto itr = result.begin(); itr != result.end(); itr++) {
          sum += itr->second.size();
        }

        STAT.add_stat(worker_id_, "leaf_task", sum);
      }

      /**
       * @brief redistribute inner node tasks for the current thread. It will
       * read @depth layer's information about node modifications and determine
       * tasks that belongs to the current thread.
       *
       * @param layer which layer's modifications are we trying to colelct
       * @param cur_mods the collected tasks will be stored in @cur_mods
       */
      void redistribute_inner_tasks(int layer, NodeModsMapType &cur_mods) {
        cur_mods = node_mods_[layer];

        // discard
        for (int i = 0; i < worker_id_; i++) {
          auto &wthread = palmtree_->workers_[i];
          for (auto other_itr = wthread.node_mods_[layer].begin(); other_itr != wthread.node_mods_[layer].end(); other_itr++) {
            cur_mods.erase(other_itr->first);
          }
        }

        // Steal work from other threads
        for (int i = worker_id_+1; i < palmtree_->NUM_WORKER; i++) {
          auto &wthread = palmtree_->workers_[i];
          for (auto other_itr = wthread.node_mods_[layer].begin(); other_itr != wthread.node_mods_[layer].end(); other_itr++) {
            auto itr = cur_mods.find(other_itr->first);
            if (itr != cur_mods.end()) {
              auto &my_mods = itr->second;
              auto &other_mods = other_itr->second;
              my_mods.insert(my_mods.end(), other_mods.begin(), other_mods.end());
            }
          }
        }
      }

      /**
       * @brief carry out all operations on the tree in a serializable order,
       *  reduce operations on the same key. The result of this function is to
       *  provide proper return result for all the operations, as well as filter
       *  out the todo node modifications on the #0 layer
       *  */
      void resolve_hazards(const std::unordered_map<Node *, std::vector<TreeOp *>> &tree_ops UNUSED) {
        node_mods_[0].clear();
        auto &leaf_mods = node_mods_[0];
        std::unordered_map<KeyType, ValueType> changed_values;
        std::unordered_set<KeyType> deleted;
        for (auto itr = tree_ops.begin(); itr != tree_ops.end(); itr++) {
          LeafNode *leaf = static_cast<LeafNode *>(itr->first);
          auto &ops = itr->second;
          for (auto op : ops) {
            if (op->op_type_ == TREE_OP_FIND) {
              if (deleted.find(op->key_) != deleted.end()) {
                op->boolean_result_ = false;
              } else {
                if (changed_values.count(op->key_) != 0) {
                  op->result_ = changed_values[op->key_];
                  op->boolean_result_ = true;
                } else {
                  int idx = palmtree_->search_leaf(leaf->keys, leaf->slot_used, op->key_);
                  if (idx == -1 || !palmtree_->key_eq(leaf->keys[idx], op->key_)) {
                    // Not find
                    op->boolean_result_ = false;
                  } else {
                    op->result_ = leaf->values[idx];
                    op->boolean_result_ = true;
                  }
                }
              }
            } else if (op->op_type_ == TREE_OP_INSERT) {
              deleted.erase(op->key_);
              changed_values[op->key_] = op->value_;
              if (leaf_mods.count(leaf) == 0)
                leaf_mods.emplace(leaf, std::vector<NodeMod>());
              leaf_mods[leaf].push_back(NodeMod(*op));
            } else {
              changed_values.erase(op->key_);
              if (leaf_mods.count(leaf) == 0)
                leaf_mods.emplace(leaf, std::vector<NodeMod>());
              leaf_mods[leaf].push_back(NodeMod(*op));
            }
          }
        }

      } // End resolve_hazards

      /**
       * @brief Handle root split and re-insert orphaned keys. It may need to grow the tree height
       */
      void handle_root() {

        int root_depth = palmtree_->tree_depth_;
        std::vector<NodeMod> root_mods;
        // Collect root modifications from all threads
        for (auto &wthread : palmtree_->workers_) {
          auto itr = wthread.node_mods_[root_depth].begin();
          if (itr != wthread.node_mods_[root_depth].end()) {
            root_mods.insert(root_mods.end(), itr->second.begin(), itr->second.end());
          }
        }
        // Handle over to modify_node
        auto new_mod = palmtree_->modify_node(palmtree_->tree_root, root_mods);
        if (new_mod.type_ == MOD_TYPE_NONE) {
        } else if (new_mod.type_ == MOD_TYPE_ADD) {
          InnerNode *new_root = new InnerNode(nullptr, palmtree_->tree_root->level+1);
          palmtree_->tree_root->parent = new_root;
          palmtree_->add_item<InnerNode, Node *>(new_root, palmtree_->min_key_, palmtree_->tree_root);
          for (auto itr = new_mod.node_items.begin(); itr != new_mod.node_items.end(); itr++) {
            itr->second->parent = new_root;
            palmtree_->add_item<InnerNode, Node *>(new_root, itr->first, itr->second);
          }
          palmtree_->tree_root = new_root;
          palmtree_->tree_depth_ += 1;
          for (auto &wthread : palmtree_->workers_) {
             wthread.node_mods_.push_back(NodeModsMapType());
          }
          palmtree_->layer_width_.emplace_back(new std::atomic<int>(1));
        }
        // Merge root if neccessary
        while (palmtree_->tree_depth_ >= 2 && palmtree_->tree_root->slot_used == 1) {
          // Decrease root height
          auto old_root = static_cast<InnerNode *>(palmtree_->tree_root);
          palmtree_->tree_root = old_root->values[0];
          delete old_root;
          palmtree_->tree_depth_ -= 1;
          for (auto &wthread : palmtree_->workers_) {
             wthread.node_mods_.pop_back();
          }
          delete palmtree_->layer_width_.back();
          palmtree_->layer_width_.pop_back();
        }
        // Naively insert orphaned
        for (auto itr = new_mod.orphaned_kv.begin(); itr != new_mod.orphaned_kv.end(); itr++) {
          auto leaf = palmtree_->search(itr->first);
          palmtree_->add_item<LeafNode, ValueType>(leaf, itr->first, itr->second);
        }
        palmtree_->ensure_min_key();
      } // End of handle_root()

      // Worker loop: process tasks
      void worker_loop() {
        while (!done_) {
          // Stage 0, collect work batch and partition
          cycletimer::SysClock start_tick = cycletimer::currentTicks();
          collect_batch();
          if (worker_id_ == 0) {
            // Check if the tree is destroyed, we must do it before the sync point
            if (palmtree_->destroyed_) {
              for (int i = 0; i < palmtree_->NUM_WORKER; i++)
                palmtree_->workers_[i].done_ = true;
            };
          }
          cycletimer::SysClock passed = cycletimer::currentTicks() - start_tick;
          STAT.add_stat(worker_id_, "stage0", passed);
          palmtree_->sync(worker_id_);


          // Stage 1, Search for leafs
          leaf_ops_.clear();
          std::unordered_map<Node *, std::vector<TreeOp *>> collected_tasks;
          for (auto op : current_tasks_) {
            op->target_node_ = palmtree_->search(op->key_);
          }

          palmtree_->sync(worker_id_);

          // Stage 2, redistribute work, read the tree then modify, each thread
          // will handle the nodes it has searched for, except the nodes that
          // have been handled by workers whose worker_id is less than me.
          // Currently we use a unordered_map to record the ownership of tasks upon
          // certain nodes.

          redistribute_leaf_tasks(collected_tasks);
          resolve_hazards(collected_tasks);
          // Modify nodes
          auto &upper_mods = node_mods_[1];
          auto &cur_mods = node_mods_[0];
          upper_mods.clear();
          for (auto itr = cur_mods.begin() ; itr != cur_mods.end(); itr++) {
            auto node = itr->first;
            auto &mods = itr->second;
            auto upper_mod = palmtree_->modify_node(node, mods);
            // FIXME: now we have orphaned_keys
            if (upper_mod.type_ == MOD_TYPE_NONE && upper_mod.orphaned_kv.empty()) {
              continue;
            }
            if (upper_mods.find(node->parent) == upper_mods.end()) {
              upper_mods.emplace(node->parent, std::vector<NodeMod>());
            }
            upper_mods[node->parent].push_back(upper_mod);
          }

          palmtree_->sync(worker_id_);

          // Stage 3, propagate tree modifications back
          // Propagate modifications until root
          for (int layer = 1; layer <= palmtree_->tree_depth_-1; layer++) {
            // DLOG_IF(INFO, worker_id_ == 0) << "Layer #" << layer << " begin";
            NodeModsMapType cur_mods;
            redistribute_inner_tasks(layer, cur_mods);
            auto &upper_mods = node_mods_[layer+1];
            upper_mods.clear();
            for (auto itr = cur_mods.begin(); itr != cur_mods.end(); itr++) {
              auto node = itr->first;
              auto &mods = itr->second;
              auto mod_res = palmtree_->modify_node(node, mods);
              if (upper_mods.count(node->parent) == 0) {
                upper_mods.emplace(node->parent, std::vector<NodeMod>());
              }
              upper_mods[node->parent].push_back(mod_res);
            }
            palmtree_->sync(worker_id_);
          } // End propagate

          palmtree_->sync(worker_id_);

          // Stage 4, modify the root, re-insert orphaned, mark work as done
          if (worker_id_ == 0) {
            cycletimer::SysClock st = cycletimer::currentTicks();
            // Mark tasks as done
            handle_root();
            STAT.add_stat(worker_id_, "end_stage", cycletimer::currentTicks() - st);
            // palmtree_->ensure_tree_structure(palmtree_->tree_root, 0);
          }

          auto st2 = cycletimer::currentTicks();
          STAT.add_stat(worker_id_, "deliver tasks", cycletimer::currentTicks() - st2);

          auto st3 = cycletimer::currentTicks();
          palmtree_->task_nums -= current_tasks_.size();
          STAT.add_stat(worker_id_, "dec task num", cycletimer::currentTicks() - st3);

          current_tasks_.clear();
          palmtree_->sync(worker_id_);

          // Free the current batch

          if (worker_id_ == 0 && batch_id % 2 == 0 && palmtree_->current_batch_ != nullptr) {
            palmtree_->current_batch_->destroy();
            free(palmtree_->current_batch_);
            palmtree_->current_batch_ = nullptr;
          }


          cycletimer::SysClock end_tick = cycletimer::currentTicks();

          STAT.add_stat(worker_id_, "round_time", end_tick-start_tick);
        } // End worker loop
      }
    }; // End WorkerThread

    std::vector<WorkerThread> workers_;
    /**********************
     * PalmTree public    *
     * ********************/
  public:
    std::atomic<int> task_nums;

    PalmTree(KeyType min_key, int num_worker):
      tree_depth_(1),
      destroyed_(false),
      min_key_(min_key),
      barrier_(num_worker),
      task_batch_queue_{1024*500}
    {
      NUM_WORKER = num_worker;
      BATCH_SIZE = BATCH_SIZE_PER_WORKER * NUM_WORKER;

      // Init the root node
      tree_root = new InnerNode(nullptr, 1);
      add_item<InnerNode, Node *>((InnerNode *)tree_root, min_key_, new LeafNode(tree_root, 0));
      // Init layer width
      layer_width_.push_back(new std::atomic<int>(1));
      layer_width_.push_back(new std::atomic<int>(1));
      // Init current batch
      current_batch_ = nullptr;
      tree_current_batch_ = (TaskBatch *)malloc(sizeof(TaskBatch));
      new (tree_current_batch_) TaskBatch(BATCH_SIZE);
      // Init stats

      STAT = Stats(NUM_WORKER);
      STAT.init_metric("batch_sort");
      STAT.init_metric("stage0");
      STAT.init_metric("stage1");
      STAT.init_metric("redist_leaf");
      STAT.init_metric("resolve_hazards");
      STAT.init_metric("stage2");
      STAT.init_metric("stage3");
      STAT.init_metric("stage4");
      STAT.init_metric("end_stage");

      STAT.init_metric("search_inner");
      STAT.init_metric("search_leaf");

      STAT.init_metric("leaf_task");

      STAT.init_metric("sync_time");
      STAT.init_metric("round_time");

      STAT.init_metric("deliver tasks");
      STAT.init_metric("dec task num");

      // Init the worker thread
      // Init the worker thread and start them
      for (int worker_id = 0; worker_id < NUM_WORKER; worker_id++) {
        workers_.emplace_back(worker_id, this);
      }
      for (auto &worker : workers_) {
        worker.start();
      }

      task_nums = 0;
    }

    // Recursively free the resources of one tree node
    void free_recursive(Node *node UNUSED) {
      if (node->type() == INNERNODE) {
        auto ptr = (InnerNode *)node;
        for(int i = 0; i < ptr->slot_used; i++) {
          free_recursive(ptr->values[i]);
        }
      }

      delete node;
    }

    ~PalmTree() {

      // Mark the tree as destroyed
      destroyed_ = true;
      // Join all workter thread
      for (auto &wthread : workers_)
        wthread.wthread_.join();
      // Free atomic layer width
      while (!layer_width_.empty()) {
        delete layer_width_.back();
        layer_width_.pop_back();
      }

      STAT.print_stat();

      free_recursive(tree_root);


      if (tree_current_batch_ != nullptr) {
        tree_current_batch_->destroy();
        free(tree_current_batch_);
      }
    }

    /**
     * @brief execute a batch of tree operations, the batch will be executed
     *  cooperatively by all worker threads
     */
    void execute_batch(std::vector<TreeOp> &operations UNUSED) {

    }

    /**
     * @brief Find the value for a key
     * @param key the key to be retrieved
     * @return nullptr if no such k,v pair
     */
    bool find(const KeyType &key UNUSED, ValueType &value UNUSED) {
      push_task(TREE_OP_FIND, &key, nullptr);

      // op.wait();
      //if (op.boolean_result_)
        //value = op.result_;
      //return op.boolean_result_;
      return true;
    }

    /**
     * @brief insert a k,v into the tree
     */
    void insert(const KeyType &key UNUSED, const ValueType &value UNUSED) {
      // TreeOp op(TREE_OP_INSERT, key, value);

      push_task(TREE_OP_INSERT, &key, &value);

      // op.wait();
    }

    /**
     * @brief remove a k,v from the tree
     */
    void remove(const KeyType &key UNUSED) {
      push_task(TREE_OP_REMOVE, &key, nullptr);

      // op->wait();
    }

    void reset_metric() {
      STAT.reset_metric();
    }

    int batch_size() {
      return BATCH_SIZE_PER_WORKER * NUM_WORKER;
    }

    // Wait until all task finished
    void wait_finish() {
      if (tree_current_batch_->size() != 0) {
        task_batch_queue_.push(tree_current_batch_);
        tree_current_batch_ = (TaskBatch *)malloc(sizeof(TaskBatch));
        new (tree_current_batch_) TaskBatch(BATCH_SIZE);
      }
      while (task_nums != 0)
        ;
    }
  }; // End of PalmTree
  // Explicit template initialization
  template class PalmTree<int, int>;
} // End of namespace palmtree

