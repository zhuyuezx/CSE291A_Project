#pragma once
#include "Move.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <limits>

/**
 * MNode.hpp
 * Monte Carlo Tree Search Node
 */
class MNode {
public:
    Move move;
    MNode* parent;  // Raw pointer to parent (parent owns children)
    std::vector<std::unique_ptr<MNode>> children;
    
    double numWins = 0;
    int numSims = 0;
    bool isTerminal = false;
    
    MNode(const Move& m, MNode* p) : move(m), parent(p) {}
    
    bool isLeaf() const { return children.empty(); }
    bool isNew() const { return numSims == 0; }
    
    // Upper Confidence Bound for Trees
    double getUCT(double uctConst) const {
        if (parent == nullptr || parent->numSims == 0) {
            return std::numeric_limits<double>::infinity();
        }
        if (numSims == 0) {
            return std::numeric_limits<double>::infinity();
        }
        return (numWins / numSims) + uctConst * std::sqrt(std::log(parent->numSims) / numSims);
    }
    
    double getWinRate() const {
        if (numSims == 0) return 0;
        return numWins / numSims;
    }
    
    // Get child with maximum UCT value
    MNode* getMaxUCTChild(double uctConst) {
        if (children.empty()) return nullptr;
        
        double maxUCT = -std::numeric_limits<double>::infinity();
        std::vector<MNode*> maxChildren;
        
        for (auto& child : children) {
            double uct = child->getUCT(uctConst);
            if (uct > maxUCT) {
                maxUCT = uct;
                maxChildren.clear();
                maxChildren.push_back(child.get());
            } else if (uct == maxUCT) {
                maxChildren.push_back(child.get());
            }
        }
        
        // Random choice among ties
        if (maxChildren.size() == 1) {
            return maxChildren[0];
        }
        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<> dis(0, maxChildren.size() - 1);
        return maxChildren[dis(rng)];
    }
    
    // Get child with maximum win rate
    MNode* getMaxWinRateChild() {
        if (children.empty()) return nullptr;
        
        double maxRate = -std::numeric_limits<double>::infinity();
        MNode* maxChild = nullptr;
        
        for (auto& child : children) {
            double rate = child->getWinRate();
            if (rate > maxRate) {
                maxRate = rate;
                maxChild = child.get();
            }
        }
        return maxChild;
    }
    
    // Get child with maximum simulations
    MNode* getMaxSimsChild() {
        if (children.empty()) return nullptr;
        
        int maxSims = -1;
        MNode* maxChild = nullptr;
        
        for (auto& child : children) {
            if (child->numSims > maxSims) {
                maxSims = child->numSims;
                maxChild = child.get();
            }
        }
        return maxChild;
    }
    
    void addChild(std::unique_ptr<MNode> child) {
        children.push_back(std::move(child));
    }
};
