#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

struct Label {
    double priority;
    std::uint64_t order;
    int node;
    double distance;
};
struct Later {
    bool operator()(const Label& a, const Label& b) const {
        return a.priority != b.priority ? a.priority > b.priority : a.order > b.order;
    }
};

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

EXPORT int shortest_path(int n, const std::int64_t* offset, const int* next,
        const double* weight, const double* heuristic, int source, int target,
        double cutoff, int* path, double* cost, std::int64_t* explored) {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distance(n, inf);
    std::vector<int> parent(n, -1);
    std::priority_queue<Label, std::vector<Label>, Later> queue;
    std::uint64_t order = 0;
    distance[source] = 0;
    queue.push({0, order++, source, 0});
    *explored = 0;
    while (!queue.empty()) {
        const Label current = queue.top();
        queue.pop();
        ++*explored;
        if (current.distance > distance[current.node]) continue;
        if (current.node == target) {
            int length = 0;
            for (int v = target; v != -1; v = parent[v]) {
                if (length >= n) return -2;
                path[length++] = v;
            }
            std::reverse(path, path + length);
            *cost = current.distance;
            return length;
        }
        for (std::int64_t edge = offset[current.node]; edge < offset[current.node + 1]; ++edge) {
            if (weight[edge] < -1e-9) return -3;
            const int v = next[edge];
            const double candidate = current.distance + weight[edge];
            if (candidate >= distance[v]) continue;
            const double priority = candidate + (heuristic ? heuristic[v] : 0);
            if (priority >= cutoff) continue;
            distance[v] = candidate;
            parent[v] = current.node;
            queue.push({priority, order++, v, candidate});
        }
    }
    return 0;
}
