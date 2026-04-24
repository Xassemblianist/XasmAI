#pragma once
/*
 * ██╗  ██╗ █████╗ ███████╗███╗   ███╗     █████╗ ██╗
 * ╚██╗██╔╝██╔══██╗██╔════╝████╗ ████║    ██╔══██╗██║
 *  ╚███╔╝ ███████║███████╗██╔████╔██║    ███████║██║
 *  ██╔██╗ ██╔══██║╚════██║██║╚██╔╝██║    ██╔══██║██║
 * ██╔╝ ██╗██║  ██║███████║██║ ╚═╝ ██║    ██║  ██║██║
 * ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝    ╚═╝  ╚═╝██║
 *
 * XasmAI — C++20 Neural Network Framework
 * Tek include ile kullanım: #include "xasmai/xasmai.hpp"
 *
 * Örnek:
 *   auto net = xasm::Network({2, 8, 4, 1});
 *   net.train(data, {.epochs = 5000, .lr = 0.1});
 *   auto result = net.predict({1, 0});
 *
 * github.com/ostriquetrum/XasmAI
 */

#include "matrix.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "layers.hpp"
#include "network.hpp"
#include "utils.hpp"
