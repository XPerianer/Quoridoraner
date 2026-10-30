// AlphaZero pipeline engine: perspective-canonical features, batched PUCT, negamax.
// Canonical frame: the side to move always walks toward row 0. Player 1's board
// is flipped vertically; an optional left-right mirror provides augmentation.
// Policies cross the boundary in canonical move IDs (0-80 pawn, 81-144 h, 145-208 v).
#include "engine.hpp"
#include <atomic>
#include <cstring>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>
using quoridor::Board;
using Key = std::array<uint64_t, 3>;
namespace {
constexpr int PLANES = 19, AREA = 81, FEATURES = PLANES * AREA;
thread_local std::string error;
Board unpack(const uint64_t* p) {
    Board b;
    b.pawn[0] = p[2] & 127; b.pawn[1] = (p[2] >> 7) & 127;
    b.walls[0] = (p[2] >> 14) & 15; b.walls[1] = (p[2] >> 18) & 15; b.turn = (p[2] >> 22) & 1;
    if (b.pawn[0] > 80 || b.pawn[1] > 80 || b.pawn[0] == b.pawn[1] || b.walls[0] > 10 || b.walls[1] > 10 || p[2] >> 23)
        throw std::runtime_error("Invalid position");
    for (int v = 0; v < 2; ++v) for (int i = 0; i < 64; ++i) if ((p[v] >> i) & 1) {
        if (!b.fits(v, i%8, i/8)) throw std::runtime_error("Overlapping walls");
        b.setWall(v, i, true);
    }
    b.initHash(); return b;
}
Key packed(const Board& b) {
    Key k{};
    for (int i=0;i<64;i++) { k[0] |= uint64_t(b.h[i]) << i; k[1] |= uint64_t(b.v[i]) << i; }
    k[2] = b.pawn[0] | (b.pawn[1]<<7) | (b.walls[0]<<14) | (b.walls[1]<<18) | (b.turn<<22);
    return k;
}
// Real <-> canonical move IDs. Both transforms are involutions.
int canonical(int move, bool flip, bool mirror) {
    if (move < 81) {
        int x = move % 9, y = move / 9;
        if (flip) y = 8 - y;
        if (mirror) x = 8 - x;
        return y * 9 + x;
    }
    int base = move >= 145 ? 145 : 81, i = move - base, x = i % 8, y = i / 8;
    if (flip) y = 7 - y;
    if (mirror) x = 7 - x;
    return base + y * 8 + x;
}
int cell(int c, bool flip, bool mirror) { return canonical(c, flip, mirror); }
void fromPawn(const Board& b, int start, int* dist) {
    std::fill(dist, dist+81, -1);
    int queue[81], head = 0, tail = 0; dist[start] = 0; queue[tail++] = start;
    while (head < tail) {
        int c = queue[head++], x = c%9, y = c/9;
        for (int d = 0; d < 4; d++) {
            if (!b.canStep(x,y,d)) continue;
            int next = c + quoridor::DY[d]*9 + quoridor::DX[d];
            if (dist[next] < 0) { dist[next] = dist[c]+1; queue[tail++] = next; }
        }
    }
}
// Goal distances depend only on wall geometry. Share this bounded lookup across
// every game and native worker, rather than running two BFS traversals per leaf.
void goalMaps(const Board& b, int (&out)[2][81]) {
    using Geometry=std::array<uint64_t,2>;
    using Maps=std::array<std::array<int,81>,2>;
    static std::mutex mutex;
    static std::map<Geometry,Maps> cache;
    Geometry key{};
    for(int i=0;i<64;++i) { key[0]|=uint64_t(b.h[i])<<i; key[1]|=uint64_t(b.v[i])<<i; }
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it=cache.find(key);
        if(it!=cache.end()) { for(int p=0;p<2;++p) std::copy(it->second[p].begin(),it->second[p].end(),out[p]); return; }
    }
    Maps maps;
    for(int p=0;p<2;++p) { b.distanceMap(p,maps[p].data()); std::copy(maps[p].begin(),maps[p].end(),out[p]); }
    std::lock_guard<std::mutex> lock(mutex);
    if(cache.size()>=4096) cache.erase(cache.begin());
    cache.emplace(key,std::move(maps));
}
void features(const Board& b, int repetitions, float remaining, bool mirror, float* out) {
    std::fill(out, out+FEATURES, 0.0f);
    const int me = b.turn, opp = me ^ 1;
    const bool flip = me == 1;
    float* plane[PLANES]; for (int c = 0; c < PLANES; c++) plane[c] = out + c*AREA;
    plane[0][cell(b.pawn[me], flip, mirror)] = 1;
    plane[1][cell(b.pawn[opp], flip, mirror)] = 1;
    for (int i = 0; i < 64; i++) {
        // Wall anchor (x, y) occupies square (x, y) of an otherwise unused 9x9 plane.
        if (b.h[i]) { int j = canonical(81+i, flip, mirror)-81; plane[2][(j/8)*9+j%8] = 1; }
        if (b.v[i]) { int j = canonical(145+i, flip, mirror)-145; plane[3][(j/8)*9+j%8] = 1; }
    }
    // Blocked steps in canonical directions: up, down, left, right.
    for (int c = 0; c < 81; c++) {
        int x = c%9, y = c/9, t = cell(c, flip, mirror);
        int up = flip ? 2 : 0, down = flip ? 0 : 2, left = mirror ? 1 : 3, right = mirror ? 3 : 1;
        plane[4][t] = !b.canStep(x,y,up); plane[5][t] = !b.canStep(x,y,down);
        plane[6][t] = !b.canStep(x,y,left); plane[7][t] = !b.canStep(x,y,right);
    }
    int goal[2][81], from[2][81];
    goalMaps(b, goal);
    for (int p = 0; p < 2; p++) fromPawn(b, b.pawn[p], from[p]);
    const int mine = goal[me][b.pawn[me]], theirs = goal[opp][b.pawn[opp]];
    for (int c = 0; c < 81; c++) {
        int t = cell(c, flip, mirror);
        plane[8][t] = goal[me][c] < 0 ? 1.0f : goal[me][c] / 16.0f;
        plane[9][t] = goal[opp][c] < 0 ? 1.0f : goal[opp][c] / 16.0f;
        plane[10][t] = from[me][c] >= 0 && from[me][c] + goal[me][c] == mine;
        plane[11][t] = from[opp][c] >= 0 && from[opp][c] + goal[opp][c] == theirs;
    }
    const float constants[7] = {b.walls[me]/10.0f, b.walls[opp]/10.0f,
        std::max(-1.0f, std::min(1.0f, (theirs-mine)/10.0f)), float(me),
        std::min(repetitions, 3)/3.0f, remaining, 1.0f};
    for (int c = 0; c < 7; c++) std::fill(plane[12+c], plane[12+c]+AREA, constants[c]);
}
class Team {
    std::mutex mutex;
    std::condition_variable ready, finished;
    std::vector<std::thread> threads;
    std::function<void(int,int)> work;
    std::atomic<int> next{0};
    int count=0, generation=0, remaining=0;
    bool stop=false;
    std::exception_ptr failure;
    void execute(int worker) {
        try { for (int i; (i=next.fetch_add(1))<count;) work(worker,i); }
        catch (...) { std::lock_guard<std::mutex> lock(mutex); if (!failure) failure=std::current_exception(); }
    }
public:
    explicit Team(int n) {
        if (n<1 || n>256) throw std::runtime_error("Worker count must be in [1,256]");
        for (int id=1; id<n; ++id) threads.emplace_back([this,id] {
            int seen=0;
            std::unique_lock<std::mutex> lock(mutex);
            while (true) {
                ready.wait(lock,[&] { return stop || generation!=seen; });
                if (stop) return;
                seen=generation; lock.unlock(); execute(id); lock.lock();
                if (--remaining==0) finished.notify_one();
            }
        });
    }
    ~Team() {
        { std::lock_guard<std::mutex> lock(mutex); stop=true; }
        ready.notify_all(); for (auto& t:threads) if(t.joinable()) t.join();
    }
    void run(int n, std::function<void(int,int)> fn) {
        if (!n) return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            work=std::move(fn); count=n; next=0; failure=nullptr;
            remaining=threads.size(); ++generation;
        }
        ready.notify_all(); execute(0);
        std::unique_lock<std::mutex> lock(mutex);
        finished.wait(lock,[&] { return remaining==0; });
        work=nullptr; if (failure) std::rethrow_exception(failure);
    }
};
struct Node;
struct Edge { int move; float prior=0; std::unique_ptr<Node> child; };
struct Node {
    Board board;
    Key key;
    int visits=0;
    double total=0;  // From this node's side-to-move perspective.
    bool expanded=false;
    std::vector<Edge> edges;
    explicit Node(Board b): board(b), key(packed(b)) { board.sp=0; }
    void legal() {
        if (!edges.empty() || board.winner()!=-1) return;
        int moves[136], n=board.generate(moves); edges.reserve(n);
        for (int i=0;i<n;i++) edges.push_back({moves[i],0,nullptr});
    }
};
struct Settings { double cpuct=1.5, fpu=0.25; };
// Network outputs depend only on the features, which are a function of the board,
// repetition count (capped at 3) and remaining plies; this key is exact.
struct Leaf {
    Key board{}; int repetitions=0, left=0, maxPlies=0;
    bool operator==(const Leaf& o) const { return board==o.board && repetitions==o.repetitions && left==o.left && maxPlies==o.maxPlies; }
};
struct LeafHash {
    size_t operator()(const Leaf& k) const {
        uint64_t h=0x9e3779b97f4a7c15ull;
        auto mix=[&](uint64_t v) { h^=v+0x9e3779b97f4a7c15ull+(h<<6)+(h>>2); h*=0xff51afd7ed558ccdull; h^=h>>32; };
        for(auto v:k.board) mix(v);
        mix(uint64_t(k.repetitions)|uint64_t(k.left)<<8|uint64_t(k.maxPlies)<<24);
        return size_t(h);
    }
};
struct Entry { std::vector<float> priors; float value; };
// Shared across all games and threads. Priors are stored in the node's edge order
// (normalized over legal moves); first-in first-out eviction.
class Cache {
    static constexpr int SHARDS=64;
    struct Shard { std::mutex mutex; std::unordered_map<Leaf,Entry,LeafHash> map; std::deque<Leaf> order; };
    Shard shards[SHARDS];
    size_t limit=0;
    Shard& shard(const Leaf& k) { return shards[LeafHash()(k)>>58]; }
public:
    bool enabled() const { return limit>0; }
    void resize(size_t n) {
        limit=n;
        for(auto& s:shards) {
            std::lock_guard<std::mutex> lock(s.mutex);
            while(s.order.size()>(limit+SHARDS-1)/SHARDS) { s.map.erase(s.order.front()); s.order.pop_front(); }
        }
    }
    void clear() { for(auto& s:shards) { std::lock_guard<std::mutex> lock(s.mutex); s.map.clear(); s.order.clear(); } }
    bool find(const Leaf& k, Node* node, float& value);
    void insert(const Leaf& k, const Node* node, float value) {
        if(!limit) return;
        Shard& s=shard(k); std::lock_guard<std::mutex> lock(s.mutex);
        Entry e{std::vector<float>(node->edges.size()), value};
        for(size_t i=0;i<node->edges.size();i++) e.priors[i]=node->edges[i].prior;
        if(!s.map.emplace(k,std::move(e)).second) return;
        s.order.push_back(k);
        while(s.order.size()>(limit+SHARDS-1)/SHARDS) { s.map.erase(s.order.front()); s.order.pop_front(); }
    }
};
bool Cache::find(const Leaf& k, Node* node, float& value) {
    if(!limit) return false;
    Shard& s=shard(k); std::lock_guard<std::mutex> lock(s.mutex);
    auto it=s.map.find(k);
    if(it==s.map.end()) return false;
    if(it->second.priors.size()!=node->edges.size()) throw std::runtime_error("Cached policy size mismatch");
    for(size_t i=0;i<node->edges.size();i++) node->edges[i].prior=it->second.priors[i];
    node->expanded=true; value=it->second.value; return true;
}
struct Job {
    Node* root=nullptr;
    Node* pending=nullptr;
    std::vector<Node*> path;
    std::vector<std::pair<Key,int>> history;
    std::vector<double> noise;
    Settings settings;
    Cache* cache=nullptr;
    std::atomic<int64_t>* hits=nullptr;
    Leaf leaf;  // Cache key of `pending`.
    bool active=false;
    int group=0;  // Waves evaluate one group while the other's batch is on the device.
    int ply=0, maxPlies=0, simulations=0, completed=0;
    bool finished() const { return active && !pending && completed>=simulations; }
    // Leaf key for `node` at the end of the current path; true if the cache expanded it.
    bool lookup(Node* node, float& value) {
        leaf={node->key, std::min(repetitions(node),3), maxPlies-ply-int(path.size())+1, maxPlies};
        if(!cache->find(leaf,node,value)) return false;
        ++*hits; return true;
    }
    int repetitions(const Node* node) const {
        int n=0;
        for (const auto& h:history) if (h.first==node->key) { n=h.second; break; }
        for (size_t i=1;i<path.size();i++) if(path[i]->key==node->key) n++;
        return std::max(1,n);
    }
    void backup(double value) {
        for (auto it=path.rbegin();it!=path.rend();++it) { (*it)->visits++; (*it)->total+=value; value=-value; }
        completed++;
    }
    // Bounded exact proof search for pawn-only endgames. Unknown is distinct
    // from draw: the network still evaluates anything not proved within budget.
    int solve(Board b, int depth, int plyNow, std::vector<Key>& branch, int& budget) const {
        int w=b.winner();
        if(w!=-1) return w==b.turn ? 1 : -1;
        Key key=packed(b); int repeats=0;
        for(const auto& h:history) if(h.first==key) repeats=h.second;
        for(size_t i=1;i+1<path.size();++i) if(path[i]->key==key) ++repeats;
        ++repeats; // The current proof-search node is a new occurrence.
        for(const auto& k:branch) if(k==key) ++repeats;
        if(repeats>=3 || plyNow>=maxPlies) return 0;
        if(depth==0 || --budget<0) return 2;
        int moves[136], n=b.generate(moves), best=-1; bool unknown=false;
        branch.push_back(key);
        for(int i=0;i<n;++i) {
            Board child=b; child.sp=0; child.make(moves[i]);
            int value=solve(child,depth-1,plyNow+1,branch,budget);
            if(value==2) unknown=true;
            else { best=std::max(best,-value); if(best==1) break; }
        }
        branch.pop_back();
        return best==1 ? 1 : unknown ? 2 : best;
    }
    void select() {
        if (!active) return;
        if (pending) throw std::runtime_error("Missing neural evaluation");
        if (!root->expanded) {
            path={root}; root->legal(); float value;
            if (!lookup(root,value)) { pending=root; return; }
            root->visits++; root->total+=value;
        }
        while (completed<simulations) {
            Node* current=root; path={root};
            while (true) {
                int winner=current->board.winner();
                if (winner!=-1) { backup(winner==current->board.turn ? 1 : -1); break; }
                if (repetitions(current)>=3 || ply+int(path.size())-1>=maxPlies) { backup(0); break; }
                if (!current->expanded) {
                    if(current!=root && !current->board.walls[0] && !current->board.walls[1]) {
                        std::vector<Key> branch; int budget=256;
                        int value=solve(current->board,4,ply+int(path.size())-1,branch,budget);
                        if(value!=2) { backup(value); break; }
                    }
                    current->legal(); float value;
                    if (lookup(current,value)) { backup(value); break; }
                    pending=current; return;
                }
                // First-play urgency: unvisited moves look slightly worse than the parent.
                double parentQ=current->visits ? current->total/current->visits : 0;
                double fpu=parentQ-settings.fpu, scale=settings.cpuct*std::sqrt(std::max(1,current->visits));
                double best=-INFINITY; Edge* chosen=nullptr;
                for (size_t i=0;i<current->edges.size();i++) {
                    Edge& e=current->edges[i];
                    double prior=(current==root && !noise.empty()) ? 0.75*e.prior+0.25*noise[i] : e.prior;
                    int visits=e.child ? e.child->visits : 0;
                    double q=visits ? -e.child->total/visits : fpu;
                    double score=q+scale*prior/(1+visits);
                    if(score>best) { best=score; chosen=&e; }
                }
                if (!chosen) throw std::runtime_error("Empty legal policy");
                if (!chosen->child) { Board b=current->board; b.sp=0; b.make(chosen->move); chosen->child=std::make_unique<Node>(b); }
                current=chosen->child.get(); path.push_back(current);
            }
        }
    }
};
// Dirichlet root noise from a per-search seed (splitmix64; Marsaglia-Tsang gamma).
struct Random {
    uint64_t state;
    uint64_t next() { uint64_t z=(state+=0x9e3779b97f4a7c15ull); z=(z^(z>>30))*0xbf58476d1ce4e5b9ull; z=(z^(z>>27))*0x94d049bb133111ebull; return z^(z>>31); }
    double uniform() { return (next()>>11)*0x1.0p-53; }
    double normal() { return std::sqrt(-2*std::log(1-uniform()))*std::cos(2*M_PI*uniform()); }
    double gamma(double a) {
        if(a<1) return gamma(a+1)*std::pow(1-uniform(),1/a);
        double d=a-1.0/3, c=1/std::sqrt(9*d);
        while(true) {
            double x=normal(), v=1+c*x; if(v<=0) continue;
            v=v*v*v; double u=1-uniform();
            if(std::log(u)<0.5*x*x+d-d*v+d*std::log(v)) return d*v;
        }
    }
};
struct Pool {
    Team team;
    Settings settings;
    std::map<int,std::unique_ptr<Node>> roots;
    // Game history (position -> occurrences) per tree, kept current by az_advance.
    std::map<int,std::vector<std::pair<Key,int>>> histories;
    std::vector<Job> jobs;
    Cache cache;
    // Per group: jobs awaiting evaluation and their row in the deduplicated batch.
    std::vector<int> leaves[2], rows[2];
    void reset() { jobs.clear(); for(int g=0;g<2;g++) { leaves[g].clear(); rows[g].clear(); } }
    std::atomic<int64_t> hits{0};
    int64_t duplicates=0, unique=0, waves=0;
    std::vector<std::unique_ptr<quoridor::Searcher>> opponents;
    explicit Pool(int workers):team(workers),opponents(workers) {}
};
struct Writer {
    std::vector<uint8_t> bytes;
    template<class T> void put(T value) {
        auto p=reinterpret_cast<const uint8_t*>(&value); bytes.insert(bytes.end(),p,p+sizeof(T));
    }
    void node(const Node& n) {
        for(auto k:n.key) put(k);
        put(n.visits); put(n.total); put(uint8_t(n.expanded)); put(uint32_t(n.edges.size()));
        for(const auto& e:n.edges) { put(e.move); put(e.prior); put(uint8_t(bool(e.child))); if(e.child) node(*e.child); }
    }
};
struct Reader {
    const uint8_t* data; size_t size, offset=0;
    template<class T> T get() {
        if(sizeof(T)>size-offset) throw std::runtime_error("Truncated tree checkpoint");
        T value; std::memcpy(&value,data+offset,sizeof(T)); offset+=sizeof(T); return value;
    }
    std::unique_ptr<Node> node(int depth=0) {
        if(depth>512) throw std::runtime_error("Invalid checkpoint depth");
        Key key; for(auto& k:key) k=get<uint64_t>();
        auto n=std::make_unique<Node>(unpack(key.data()));
        n->visits=get<int>(); n->total=get<double>(); n->expanded=get<uint8_t>();
        auto count=get<uint32_t>(); if(count>136) throw std::runtime_error("Invalid checkpoint edges");
        for(uint32_t i=0;i<count;++i) {
            Edge e; e.move=get<int>(); e.prior=get<float>();
            if(get<uint8_t>()) e.child=node(depth+1);
            n->edges.push_back(std::move(e));
        }
        return n;
    }
};
template<class F> int guard(F fn) {
    try { error.clear(); return fn(); }
    catch(const std::exception& e) { error=e.what(); return -1; }
    catch(...) { error="Unknown native error"; return -1; }
}
}
extern "C" {
const char* az_error() { return error.c_str(); }
int az_planes() { return PLANES; }
int az_snapshot(Pool* p, uint8_t* out, int capacity) {
    return guard([&] {
        Writer w; w.put(uint32_t(1)); w.put(uint32_t(p->roots.size()));
        for(const auto& item:p->roots) { w.put(item.first); w.node(*item.second); }
        if(w.bytes.size()>size_t(INT32_MAX)) throw std::runtime_error("Tree checkpoint too large");
        if(out) {
            if(capacity<int(w.bytes.size())) throw std::runtime_error("Tree buffer too small");
            std::copy(w.bytes.begin(),w.bytes.end(),out);
        }
        return int(w.bytes.size());
    });
}
int az_restore(Pool* p, const uint8_t* data, int size) {
    return guard([&] {
        if(size<0) throw std::runtime_error("Invalid tree size");
        Reader r{data,size_t(size)};
        if(r.get<uint32_t>()!=1) throw std::runtime_error("Unknown tree format");
        auto count=r.get<uint32_t>();
        std::map<int,std::unique_ptr<Node>> roots;
        for(uint32_t i=0;i<count;++i) { int id=r.get<int>(); roots[id]=r.node(); }
        if(r.offset!=r.size) throw std::runtime_error("Trailing tree data");
        p->reset(); p->roots=std::move(roots); p->histories.clear(); return 0;
    });
}

void* az_create(int workers) { try { error.clear(); return new Pool(workers); } catch(const std::exception& e) { error=e.what(); return nullptr; } }
void az_destroy(Pool* p) { delete p; }
int az_settings(Pool* p, double cpuct, double fpu) { return guard([&] { p->settings={cpuct,fpu}; return 0; }); }
int az_clear(Pool* p) { return guard([&] { p->reset(); p->roots.clear(); p->histories.clear(); return 0; }); }
int az_drop(Pool* p,int id) { return guard([&] { p->roots.erase(id); p->histories.erase(id); return 0; }); }
int az_history(Pool* p,int id,const uint64_t* keys,const int32_t* counts,int n) {
    return guard([&] {
        auto& h=p->histories[id]; h.clear();
        for(int i=0;i<n;i++) h.emplace_back(Key{keys[3*i],keys[3*i+1],keys[3*i+2]},counts[i]);
        return 0;
    });
}
// Legal real moves; returns count, or 0 for a finished game.
int az_legal(const uint64_t* position,int32_t* legal) {
    return guard([&] { Board b=unpack(position); return b.winner()==-1?b.generate(legal):0; });
}
int az_distances(const uint64_t* position,int32_t* out) {
    return guard([&] { Board b=unpack(position); out[0]=b.distance(0); out[1]=b.distance(1); return 0; });
}
// Apply real moves; for ids >= 0, keeps the matching subtree as the new root and counts
// the new position in that tree's history.
int az_advance(Pool* p,int count,const int32_t* ids,const uint64_t* positions,const int32_t* moves,uint64_t* results) {
    return guard([&] {
        for(int i=0;i<count;i++) {
            Board b=unpack(positions+3*i); int legal[136], n=b.winner()==-1?b.generate(legal):0, move=moves[i];
            if(std::find(legal,legal+n,move)==legal+n) throw std::runtime_error("Illegal move");
            Key before=packed(b); b.make(move); Key after=packed(b); std::copy(after.begin(),after.end(),results+3*i);
            if(ids[i]<0) continue;
            auto history=p->histories.find(ids[i]);
            if(history==p->histories.end()) throw std::runtime_error("Unknown game history");
            auto seen=std::find_if(history->second.begin(),history->second.end(),[&](const auto& h) { return h.first==after; });
            if(seen==history->second.end()) history->second.emplace_back(after,1); else seen->second++;
            auto& root=p->roots[ids[i]]; std::unique_ptr<Node> next;
            if(root && root->key==before) for(auto& e:root->edges) if(e.move==move) { next=std::move(e.child); break; }
            root=next?std::move(next):std::make_unique<Node>(b);
        }
        return 0;
    });
}
// Batch feature extraction for supervised data.
int az_features(Pool* p,const uint64_t* positions,const int32_t* repetitions,const float* remaining,
                const uint8_t* mirror,int n,float* out) {
    return guard([&] {
        p->team.run(n,[&](int,int i) { features(unpack(positions+3*i),repetitions[i],remaining[i],mirror[i],out+size_t(i)*FEATURES); });
        return 0;
    });
}
// Map 209-wide policy rows between real and canonical IDs (involution).
int az_canonical_moves(int turn,int mirror,int32_t* out) {
    return guard([&] { for(int m=0;m<209;m++) out[m]=canonical(m,turn==1,mirror); return 0; });
}
int az_start(Pool* p) { return guard([&] { p->reset(); return 0; }); }
// Searches the position of tree `id` against its registered history. With `noise`, the
// root priors mix in Dirichlet(0.3) noise drawn from `seed`.
int az_begin(Pool* p,int id,const uint64_t* position,int ply,int maxPlies,int simulations,uint64_t seed,int noise,int group) {
    return guard([&] {
        if(simulations<1 || ply<0 || maxPlies<=ply) throw std::runtime_error("Invalid search budget");
        if(group<0 || group>1) throw std::runtime_error("Invalid search group");
        Board b=unpack(position); Key k=packed(b); auto& root=p->roots[id];
        if(!root || root->key!=k) root=std::make_unique<Node>(b);
        Job j; j.root=root.get(); j.ply=ply; j.maxPlies=maxPlies; j.simulations=simulations; j.settings=p->settings;
        j.cache=&p->cache; j.hits=&p->hits; j.active=true; j.group=group;
        // Retained visits count toward the budget; the root's own expansion is not a simulation.
        j.completed=std::min(simulations-1,std::max(0,root->visits-1));
        auto history=p->histories.find(id);
        if(history==p->histories.end()) throw std::runtime_error("Unknown game history");
        j.history=history->second;
        if(b.winner()!=-1 || j.repetitions(root.get())>=3) throw std::runtime_error("Cannot search a finished game");
        root->legal();
        if(noise) {
            Random r{seed}; double sum=0;
            for(size_t i=0;i<root->edges.size();i++) { j.noise.push_back(r.gamma(0.3)); sum+=j.noise.back(); }
            for(auto& v:j.noise) v/=sum;
        }
        for(size_t i=0;i<p->jobs.size();i++) if(!p->jobs[i].active) { p->jobs[i]=std::move(j); return int(i); }
        p->jobs.push_back(std::move(j)); return int(p->jobs.size())-1;
    });
}
// Advances every active job of `group` to its next uncached leaf. Writes features and
// canonical legal move IDs for each distinct leaf (identical leaves share one row);
// returns the row count. Zero means every active job of the group has finished its budget.
int az_wave(Pool* p,int group,float* out,int32_t* legal,int32_t* counts) {
    return guard([&] {
        if(group<0 || group>1) throw std::runtime_error("Invalid search group");
        if(!p->leaves[group].empty()) throw std::runtime_error("Group still awaits evaluation");
        p->team.run(p->jobs.size(),[&](int,int i){ if(p->jobs[i].group==group) p->jobs[i].select(); });
        auto& leaves=p->leaves[group]; auto& rows=p->rows[group];
        std::unordered_map<Leaf,int,LeafHash> seen;
        std::vector<int> first;
        for(size_t i=0;i<p->jobs.size();i++) {
            if(!p->jobs[i].active || p->jobs[i].group!=group || !p->jobs[i].pending) continue;
            int row=int(first.size());
            if(p->cache.enabled()) {
                auto it=seen.emplace(p->jobs[i].leaf,row);
                if(!it.second) { row=it.first->second; ++p->duplicates; }
            }
            if(row==int(first.size())) first.push_back(i);
            leaves.push_back(i); rows.push_back(row);
        }
        p->team.run(first.size(),[&](int,int n) {
            Job& j=p->jobs[first[n]]; const Board& b=j.pending->board;
            const int depth=int(j.path.size())-1;
            counts[n]=j.pending->edges.size();
            features(b,j.repetitions(j.pending),float(j.maxPlies-j.ply-depth)/j.maxPlies,false,out+size_t(n)*FEATURES);
            std::fill(legal+n*136,legal+(n+1)*136,0);
            for(size_t k=0;k<j.pending->edges.size();k++) legal[n*136+k]=canonical(j.pending->edges[k].move,b.turn==1,false);
        });
        p->unique+=first.size(); if(!first.empty()) ++p->waves;
        return int(first.size());
    });
}
// Rows follow az_wave's order for the same group; every job sharing a row receives the same evaluation.
int az_evaluate(Pool* p,int group,const float* policies,const float* values,int n) {
    return guard([&] {
        if(group<0 || group>1) throw std::runtime_error("Invalid search group");
        auto& leaves=p->leaves[group]; auto& rows=p->rows[group];
        std::vector<char> stored(n,0);
        for(size_t i=0;i<leaves.size();i++) {
            int r=rows[i];
            if(r<0 || r>=n) throw std::runtime_error("Missing leaf evaluation");
            Job& j=p->jobs[leaves[i]]; if(!j.pending) throw std::runtime_error("Unexpected evaluation");
            if(!std::isfinite(values[r])) throw std::runtime_error("Invalid value");
            double sum=0;
            for(size_t k=0;k<j.pending->edges.size();k++) { double v=policies[r*136+k]; if(!std::isfinite(v) || v<0) throw std::runtime_error("Invalid policy"); sum+=v; }
            if(sum<=0) throw std::runtime_error("Empty policy");
            for(size_t k=0;k<j.pending->edges.size();k++) j.pending->edges[k].prior=policies[r*136+k]/sum;
            j.pending->expanded=true;
            if(!stored[r]) { p->cache.insert(j.leaf,j.pending,values[r]); stored[r]=1; }
            if(j.pending==j.root && j.path.size()==1) { j.root->visits++; j.root->total+=values[r]; }
            else j.backup(values[r]);
            j.pending=nullptr;
        }
        leaves.clear(); rows.clear();
        return 0;
    });
}
// Slots whose search budget is complete (written in ascending order); returns the count.
int az_finished(Pool* p,int32_t* out) {
    return guard([&] {
        int n=0;
        for(size_t i=0;i<p->jobs.size();i++) if(p->jobs[i].finished()) out[n++]=i;
        return n;
    });
}
// Root visit distributions (REAL move IDs) and root Q of finished slots; frees the slots.
int az_take(Pool* p,const int32_t* slots,int n,float* out,float* rootValues) {
    return guard([&] {
        for(int i=0;i<n;i++) {
            int slot=slots[i];
            if(slot<0 || slot>=int(p->jobs.size()) || !p->jobs[slot].finished()) throw std::runtime_error("Search still in progress");
            Job& j=p->jobs[slot]; float* row=out+209*size_t(i);
            double sum=0; for(auto& e:j.root->edges) if(e.child) sum+=e.child->visits;
            std::fill(row,row+209,0);
            for(auto& e:j.root->edges) if(e.child && sum>0) row[e.move]=e.child->visits/sum;
            rootValues[i]=j.root->visits ? j.root->total/j.root->visits : 0;
            j=Job();
        }
        return 0;
    });
}
int az_cache(Pool* p,int64_t limit) { return guard([&] { if(limit<0) throw std::runtime_error("Invalid cache size"); p->cache.resize(limit); return 0; }); }
int az_cache_clear(Pool* p) { return guard([&] { p->cache.clear(); return 0; }); }
// Cumulative: cache hits during selection, in-wave duplicates, distinct evaluations, waves.
int az_stats(Pool* p,int64_t* out) {
    return guard([&] { out[0]=p->hits; out[1]=p->duplicates; out[2]=p->unique; out[3]=p->waves; return 0; });
}
int az_negamax(Pool* p,const uint64_t* positions,int n,const int32_t* nodes,int64_t* out) {
    return guard([&] {
        if(n<0 || std::any_of(nodes,nodes+n,[](int v) { return v<1; })) throw std::runtime_error("Invalid negamax budget");
        p->team.run(n,[&](int worker,int i) {
            if(!p->opponents[worker]) p->opponents[worker]=std::make_unique<quoridor::Searcher>();
            auto r=p->opponents[worker]->search(unpack(positions+3*i),nodes[i]);
            out[3*i]=r.move; out[3*i+1]=r.nodes; out[3*i+2]=r.score;
        }); return 0;
    });
}
}
