#include <node_api.h>
#include <cmath>
#include <string>
#include "engine.hpp"
using quoridor::Board;

static void check(napi_status status) { if (status != napi_ok) throw std::runtime_error("Invalid native engine argument"); }
static napi_value get(napi_env env, napi_value obj, const char* name) {
    napi_value out; check(napi_get_named_property(env,obj,name,&out)); return out;
}
static int integer(napi_env env, napi_value value, int low, int high) {
    double d; check(napi_get_value_double(env,value,&d));
    if (!std::isfinite(d) || d != std::floor(d) || d < low || d > high) throw std::runtime_error("Integer argument out of range");
    return static_cast<int>(d);
}
static std::vector<int> array(napi_env env, napi_value value, int maxLength, int low, int high) {
    bool isArray; check(napi_is_array(env,value,&isArray)); if (!isArray) throw std::runtime_error("Expected array");
    uint32_t length; check(napi_get_array_length(env,value,&length));
    if (length > static_cast<uint32_t>(maxLength)) throw std::runtime_error("Array too long");
    std::vector<int> result; result.reserve(length);
    for (uint32_t i = 0; i < length; i++) { napi_value item; check(napi_get_element(env,value,i,&item)); result.push_back(integer(env,item,low,high)); }
    return result;
}
static Board readBoard(napi_env env, napi_value value) {
    Board b;
    auto pawns = array(env,get(env,value,"pawns"),2,0,80);
    auto walls = array(env,get(env,value,"wallsLeft"),2,0,10);
    if (pawns.size() != 2 || walls.size() != 2) throw std::runtime_error("Expected two players");
    for (int p = 0; p < 2; p++) { b.pawn[p] = pawns[p]; b.walls[p] = walls[p]; }
    b.turn = integer(env,get(env,value,"turn"),0,1);
    for (bool vertical : {false,true}) {
        auto slots = array(env,get(env,value,vertical ? "vWalls" : "hWalls"),64,0,63);
        for (int i : slots) {
            if (!b.fits(vertical,i%8,i/8)) throw std::runtime_error("Overlapping walls");
            b.setWall(vertical,i,true);
        }
    }
    return b;
}
static napi_value number(napi_env env, double n) { napi_value out; check(napi_create_double(env,n,&out)); return out; }
static napi_value object(napi_env env) { napi_value out; check(napi_create_object(env,&out)); return out; }
static void set(napi_env env, napi_value obj, const char* name, napi_value value) { check(napi_set_named_property(env,obj,name,value)); }
static napi_value list(napi_env env, const int* values, int n) {
    napi_value out; check(napi_create_array_with_length(env,n,&out));
    for (int i = 0; i < n; i++) check(napi_set_element(env,out,i,number(env,values[i])));
    return out;
}
static napi_value position(napi_env env, const Board& b) {
    auto out = object(env);
    set(env,out,"pawns",list(env,b.pawn,2)); set(env,out,"wallsLeft",list(env,b.walls,2)); set(env,out,"turn",number(env,b.turn));
    for (bool vertical : {false,true}) {
        int slots[64], n = 0; const auto& walls = vertical ? b.v : b.h;
        for (int i = 0; i < 64; i++) if (walls[i]) slots[n++] = i;
        set(env,out,vertical ? "vWalls" : "hWalls",list(env,slots,n));
    }
    return out;
}
template<typename F> static napi_value guard(napi_env env, F fn) {
    try { return fn(); } catch (const std::exception& e) { napi_throw_error(env,nullptr,e.what()); return nullptr; }
}
static napi_value inspect(napi_env env, napi_callback_info info) {
    return guard(env,[&] {
        size_t argc = 1; napi_value args[1]; check(napi_get_cb_info(env,info,&argc,args,nullptr,nullptr));
        if (argc != 1) throw std::runtime_error("Expected position");
        Board b = readBoard(env,args[0]); b.initHash();
        auto out = object(env); int moves[136], n = b.winner() < 0 ? b.generate(moves) : 0;
        set(env,out,"legal",list(env,moves,n)); set(env,out,"winner",number(env,b.winner()));
        int distances[2] = {b.distance(0),b.distance(1)};
        set(env,out,"distances",list(env,distances,2));
        set(env,out,"hash1",number(env,b.hash1)); set(env,out,"hash2",number(env,b.hash2));
        return out;
    });
}
static napi_value legal(napi_env env, napi_callback_info info) {
    return guard(env,[&] {
        size_t argc = 1; napi_value args[1]; check(napi_get_cb_info(env,info,&argc,args,nullptr,nullptr));
        if (argc != 1) throw std::runtime_error("Expected position");
        Board b = readBoard(env,args[0]); int moves[136], n = b.winner() < 0 ? b.generate(moves) : 0;
        return list(env,moves,n);
    });
}
static napi_value step(napi_env env, napi_callback_info info) {
    return guard(env,[&] {
        size_t argc = 3; napi_value args[3]; check(napi_get_cb_info(env,info,&argc,args,nullptr,nullptr));
        if (argc < 2) throw std::runtime_error("Expected position and move");
        Board b = readBoard(env,args[0]); int move = integer(env,args[1],0,208);
        bool validate = true;
        if (argc == 3) check(napi_get_value_bool(env,args[2],&validate));
        if (validate) {
            int moves[136], n = b.winner() < 0 ? b.generate(moves) : 0;
            if (std::find(moves,moves+n,move) == moves+n) throw std::runtime_error("Illegal move");
        }
        // Unchecked transitions are internal MCTS moves drawn from a legal-move list.
        if (move >= 81 && b.walls[b.turn] == 0) throw std::runtime_error("No walls left");
        b.make(move); return position(env,b);
    });
}
static napi_value search(napi_env env, napi_callback_info info) {
    return guard(env,[&] {
        size_t argc = 3; napi_value args[3]; check(napi_get_cb_info(env,info,&argc,args,nullptr,nullptr));
        if (argc < 2) throw std::runtime_error("Expected position and node budget");
        Board b = readBoard(env,args[0]); int nodes = integer(env,args[1],1,1000000000);
        thread_local quoridor::Searcher engine;
        auto result = engine.search(b,nodes,argc == 3 ? integer(env,args[2],1,64) : 64); auto out = object(env);
        set(env,out,"move",number(env,result.move)); set(env,out,"score",number(env,result.score));
        set(env,out,"nodes",number(env,result.nodes)); return out;
    });
}
static napi_value init(napi_env env, napi_value exports) {
    napi_property_descriptor methods[] = {
        {"inspect",nullptr,inspect,nullptr,nullptr,nullptr,napi_default,nullptr},
        {"legal",nullptr,legal,nullptr,nullptr,nullptr,napi_default,nullptr},
        {"step",nullptr,step,nullptr,nullptr,nullptr,napi_default,nullptr},
        {"search",nullptr,search,nullptr,nullptr,nullptr,napi_default,nullptr},
    };
    check(napi_define_properties(env,exports,4,methods)); return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, init)
