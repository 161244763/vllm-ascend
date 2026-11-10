#include "../src/position_queue.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <iostream>
#include <random>

static void check(const PositionQueue& q, const std::deque<int64_t>& oracle) {
    assert(q.size() == oracle.size());
    assert(q.empty() == oracle.empty());
    assert(std::equal(q.begin(), q.end(), oracle.begin(), oracle.end()));
    if (!oracle.empty()) {
        assert(q.front() == oracle.front() && q.back() == oracle.back());
        for (size_t i=0; i<oracle.size(); ++i) assert(q[i] == oracle[i]);
        auto needle = oracle[oracle.size()/2];
        assert(std::lower_bound(q.begin(), q.end(), needle) - q.begin() ==
               std::lower_bound(oracle.begin(), oracle.end(), needle) - oracle.begin());
    }
}

int main() {
    std::mt19937 rng(20261109);
    PositionQueue q;
    std::deque<int64_t> oracle;
    int64_t next = 0;
    for (int i=0; i<300000; ++i) {
        if (oracle.empty() || (oracle.size()<256 && rng()%2)) {
            q.push_back(++next); oracle.push_back(next);
        } else { q.pop_front(); oracle.pop_front(); }
        check(q, oracle);
        if (i%127 == 0) {
            auto copy=q; check(copy, oracle);
            auto moved=std::move(copy); check(moved, oracle);
        }
    }
    while (!oracle.empty()) { q.pop_front(); oracle.pop_front(); check(q,oracle); }
    std::cout << "300000 operations and copy/move/drain passed\n";
}
