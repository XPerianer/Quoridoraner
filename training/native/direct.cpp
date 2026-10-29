// In-process MCTS and game rules. No Python objects or IPC in a search wave.
#include "engine.hpp"
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
using quoridor::Board;
using Key = std::array<uint64_t, 3>;
namespace {
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
// Persistent threads amortize scheduling over thousands of tiny leaf waves.
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
        try {
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
        } catch (...) { shutdown(); throw; }
    }
    void shutdown() {
        { std::lock_guard<std::mutex> lock(mutex); stop=true; }
        ready.notify_all(); for (auto& t:threads) if(t.joinable()) t.join();
    }
    ~Team() { shutdown(); }
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
struct Edge { int move; double prior=0; std::unique_ptr<Node> child; };
struct Node {
    Board board;
    Key key;
    int visits=0;
    double total=0;
    bool expanded=false;
    std::vector<Edge> edges;
    explicit Node(Board b): board(b), key(packed(b)) { board.sp=0; }
    void legal() {
        if (!edges.empty() || board.winner()!=-1) return;
        int moves[136], n=board.generate(moves); edges.reserve(n);
        for (int i=0;i<n;i++) edges.push_back({moves[i],0,nullptr});
    }
};
struct Job {
    Node* root=nullptr;
    Node* pending=nullptr;
    std::vector<Node*> path;
    std::vector<std::pair<Key,int>> history;
    std::vector<double> noise;
    int ply=0, maxPlies=0, simulations=0, completed=0, initialVisits=0;
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
    void select() {
        if (pending) throw std::runtime_error("Missing neural evaluation");
        if (!root->expanded) { path.clear(); pending=root; root->legal(); return; }
        while (completed<simulations) {
            Node* current=root; path={root};
            while (true) {
                int winner=current->board.winner();
                if (winner!=-1) { backup(winner==current->board.turn ? 1 : -1); break; }
                if (repetitions(current)>=3 || ply+int(path.size())-1>=maxPlies) { backup(0); break; }
                if (!current->expanded) { pending=current; current->legal(); return; }
                double best=-INFINITY; Edge* chosen=nullptr;
                for (size_t i=0;i<current->edges.size();i++) {
                    Edge& e=current->edges[i];
                    double prior=(current==root && !noise.empty()) ? 0.75*e.prior+0.25*noise[i] : e.prior;
                    int visits=e.child ? e.child->visits : 0;
                    double q=visits ? -e.child->total/visits : 0;
                    double score=q+1.5*prior*std::sqrt(std::max(1,current->visits))/(1+visits);
                    if(score>best) { best=score; chosen=&e; }
                }
                if (!chosen) throw std::runtime_error("Empty legal policy");
                if (!chosen->child) { Board b=current->board; b.sp=0; b.make(chosen->move); chosen->child=std::make_unique<Node>(b); }
                current=chosen->child.get(); path.push_back(current);
            }
        }
    }
};
struct Pool {
    Team team;
    std::map<int,std::unique_ptr<Node>> roots;
    std::vector<Job> jobs;
    std::vector<std::unique_ptr<quoridor::Searcher>> opponents;
    explicit Pool(int workers):team(workers),opponents(workers) {}
};
template<class F> int guard(F fn) {
    try { error.clear(); return fn(); }
    catch(const std::exception& e) { error=e.what(); return -1; }
    catch(...) { error="Unknown native error"; return -1; }
}
void features(const Job& job, float* out) {
    std::fill(out,out+729,0.0f);
    const Board& b=job.pending->board;
    out[b.pawn[0]]=1; out[81+b.pawn[1]]=1;
    for(int i=0;i<64;i++) { int cell=(i/8)*9+i%8; out[162+cell]=b.h[i]; out[243+cell]=b.v[i]; }
    const int depth=job.path.empty()?0:int(job.path.size())-1;
    const float constants[5]={float(b.walls[0]/10.0),float(b.walls[1]/10.0),float(b.turn),
        float(std::min(job.repetitions(job.pending),3)/3.0),float((job.maxPlies-job.ply-depth)/double(job.maxPlies))};
    for(int c=0;c<5;c++) std::fill(out+(c+4)*81,out+(c+5)*81,constants[c]);
}
}
extern "C" {
const char* q_error() { return error.c_str(); }
void* q_create(int workers) { try { error.clear(); return new Pool(workers); } catch(const std::exception& e) { error=e.what(); return nullptr; } }
void q_destroy(Pool* p) { delete p; }
int q_clear(Pool* p) { return guard([&] { p->jobs.clear(); p->roots.clear(); return 0; }); }
int q_drop(Pool* p,int id) { return guard([&] { p->roots.erase(id); return 0; }); }
int q_describe(const uint64_t* position,int32_t* legal) {
    return guard([&] { Board b=unpack(position); return b.winner()==-1?b.generate(legal):0; });
}
int q_advance(Pool* p,int id,const uint64_t* position,int move,uint64_t* result) {
    return guard([&] {
        Board b=unpack(position); int moves[136], n=b.winner()==-1?b.generate(moves):0;
        if(std::find(moves,moves+n,move)==moves+n) throw std::runtime_error("Illegal move");
        Key before=packed(b); b.make(move); Key after=packed(b); std::copy(after.begin(),after.end(),result);
        if(id<0) return 0;
        auto& root=p->roots[id]; std::unique_ptr<Node> next;
        if(root && root->key==before) for(auto& e:root->edges) if(e.move==move) { next=std::move(e.child); break; }
        root=next?std::move(next):std::make_unique<Node>(b); return 0;
    });
}
int q_start(Pool* p) { return guard([&] { p->jobs.clear(); return 0; }); }
int q_begin(Pool* p,int id,const uint64_t* position,const uint64_t* history,const int32_t* counts,int historySize,
            int ply,int maxPlies,int simulations,const double* noise,int noiseSize) {
    return guard([&] {
        if(simulations<1 || ply<0 || maxPlies<=ply || historySize<0 || noiseSize<0 || noiseSize>136) throw std::runtime_error("Invalid search budget");
        Board b=unpack(position); Key k=packed(b); auto& root=p->roots[id];
        if(!root || root->key!=k) root=std::make_unique<Node>(b);
        Job j; j.root=root.get(); j.ply=ply; j.maxPlies=maxPlies; j.simulations=simulations; j.initialVisits=root->visits;
        for(int i=0;i<historySize;i++) { Key h={history[3*i],history[3*i+1],history[3*i+2]}; j.history.emplace_back(h,counts[i]); }
        if(b.winner()!=-1 || j.repetitions(root.get())>=3) throw std::runtime_error("Cannot search a finished game");
        j.noise.assign(noise,noise+noiseSize);
        root->legal();
        if(noiseSize && noiseSize!=int(root->edges.size())) throw std::runtime_error("Incorrect root noise length");
        p->jobs.push_back(std::move(j)); return 0;
    });
}
int q_wave(Pool* p,float* out,int32_t* legal,int32_t* counts,int32_t* indices) {
    return guard([&] {
        p->team.run(p->jobs.size(),[&](int,int i){ p->jobs[i].select(); });
        int n=0;
        for(size_t i=0;i<p->jobs.size();i++) {
            auto& j=p->jobs[i]; if(!j.pending) continue;
            indices[n]=i; counts[n]=j.pending->edges.size(); features(j,out+n*729);
            std::fill(legal+n*136,legal+(n+1)*136,0);
            for(size_t k=0;k<j.pending->edges.size();k++) legal[n*136+k]=j.pending->edges[k].move;
            n++;
        }
        return n;
    });
}
int q_evaluate(Pool* p,const float* policies,const float* values,const int32_t* indices,int n) {
    return guard([&] {
        for(int i=0;i<n;i++) {
            if(indices[i]<0 || indices[i]>=int(p->jobs.size())) throw std::runtime_error("Invalid leaf index");
            Job& j=p->jobs[indices[i]]; if(!j.pending) throw std::runtime_error("Unexpected evaluation");
            if(!std::isfinite(values[i])) throw std::runtime_error("Invalid value");
            double sum=0;
            for(size_t k=0;k<j.pending->edges.size();k++) { double v=policies[i*136+k]; if(!std::isfinite(v) || v<0) throw std::runtime_error("Invalid policy"); sum+=v; }
            if(sum<=0) throw std::runtime_error("Empty policy");
            for(size_t k=0;k<j.pending->edges.size();k++) j.pending->edges[k].prior=policies[i*136+k]/sum;
            j.pending->expanded=true; if(!j.path.empty()) j.backup(values[i]); j.pending=nullptr;
        }
        return 0;
    });
}
int q_policy(Pool* p,float* out,int64_t* reused) {
    return guard([&] {
        *reused=0;
        for(size_t i=0;i<p->jobs.size();i++) {
            Job& j=p->jobs[i]; if(j.pending || j.completed!=j.simulations) throw std::runtime_error("Search still in progress");
            double sum=0; for(auto& e:j.root->edges) if(e.child) sum+=e.child->visits;
            std::fill(out+i*209,out+(i+1)*209,0);
            for(auto& e:j.root->edges) if(e.child) out[i*209+e.move]=e.child->visits/sum;
            *reused+=j.initialVisits;
        }
        return 0;
    });
}
int q_negamax(Pool* p,const uint64_t* positions,int n,int nodes,int64_t* out) {
    return guard([&] {
        if(nodes<1 || n<0) throw std::runtime_error("Invalid negamax budget");
        p->team.run(n,[&](int worker,int i) {
            if(!p->opponents[worker]) p->opponents[worker]=std::make_unique<quoridor::Searcher>();
            auto r=p->opponents[worker]->search(unpack(positions+3*i),nodes);
            out[3*i]=r.move; out[3*i+1]=r.nodes; out[3*i+2]=r.score;
        }); return 0;
    });
}
// Checkpoint-only tree access; the hot path never crosses this object interface.
void* q_root(Pool* p,int id) { auto it=p->roots.find(id); return it==p->roots.end()?nullptr:it->second.get(); }
int q_inspect(Node* node,uint64_t* position,double* stats,int32_t* moves,double* priors,void** children) {
    return guard([&] {
        std::copy(node->key.begin(),node->key.end(),position); stats[0]=node->visits; stats[1]=node->total; stats[2]=node->expanded;
        for(size_t i=0;i<node->edges.size();i++) { moves[i]=node->edges[i].move; priors[i]=node->edges[i].prior; children[i]=node->edges[i].child.get(); }
        return int(node->edges.size());
    });
}
void* q_restore_node(Pool* p,int id,Node* parent,int move,const uint64_t* position,int visits,double total,int expanded,
                     const int32_t* moves,const double* priors,int n) {
    try {
        error.clear();
        if(n<0 || n>136 || visits<0 || !std::isfinite(total)) throw std::runtime_error("Invalid snapshot");
        auto next=std::make_unique<Node>(unpack(position)); auto ptr=next.get();
        next->visits=visits; next->total=total; next->expanded=expanded;
        for(int i=0;i<n;i++) { if(moves[i]<0 || moves[i]>208 || !std::isfinite(priors[i])) throw std::runtime_error("Invalid snapshot edge"); next->edges.push_back({moves[i],priors[i],nullptr}); }
        if(!parent) p->roots[id]=std::move(next);
        else { bool found=false; for(auto& e:parent->edges) if(e.move==move) { e.child=std::move(next); found=true; break; } if(!found) throw std::runtime_error("Missing parent edge"); }
        return ptr;
    } catch(const std::exception& e) { error=e.what(); return nullptr; }
}
}
