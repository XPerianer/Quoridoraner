// Separate evaluation-only library; the production trainer keeps az.cpp unchanged.
#include "az.cpp"

extern "C" int az_stop_puct(Pool* p,int slot,int32_t* completed) {
    return guard([&] {
        if(slot<0 || slot>=int(p->jobs.size())) throw std::runtime_error("Unknown timed search");
        Job& j=p->jobs[slot];
        if(!j.active || j.settings.gumbel) throw std::runtime_error("Timed stop requires active PUCT");
        if(j.pending) throw std::runtime_error("Collect pending evaluation before timed stop");
        bool visited=false;
        for(const auto& edge:j.root->edges) if(edge.child && edge.child->visits>0) visited=true;
        if(!visited) return 0;  // A root expansion alone cannot select a legal visited move.
        *completed=j.completed;
        j.simulations=j.completed;
        return 1;
    });
}
