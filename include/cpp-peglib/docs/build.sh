source ~/Projects/emsdk/emsdk_env.sh
emcc -std=c++17 -O3 --bind -o native.js native.cpp
