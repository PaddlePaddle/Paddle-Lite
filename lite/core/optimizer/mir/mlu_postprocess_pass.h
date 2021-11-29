#include <set>

  void GatherAndModifyFirstConvNodes(SSAGraph* graph);

  bool IsFirstConvNode(Node* arg_node);

  bool IsFirstConvInSubgraph(Node* arg_node, Node* inst);

  void AdjustSubgraph(Node* subgraph_node, const Type* op_type);

 private:
  std::set<std::string> first_conv_nodes_;
