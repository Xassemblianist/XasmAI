#pragma once

#include "matrix.hpp"
#include "optimizer.hpp"
#include <cmath>

namespace xasm {

class LSTMCell {
    size_t input_dim;
    size_t hidden_dim;

    Matrix Wf, Wi, Wc, Wo;
    Matrix Uf, Ui, Uc, Uo;
    Matrix bf, bi, bc, bo;

    Matrix h_prev, c_prev;

    Matrix ft, it, ct_bar, ot;
    Matrix c_t, h_t;
    Matrix x_cache;

    Matrix dWf, dWi, dWc, dWo;
    Matrix dUf, dUi, dUc, dUo;
    Matrix dbf, dbi, dbc, dbo;

    ParamState sWf, sWi, sWc, sWo;
    ParamState sUf, sUi, sUc, sUo;
    ParamState sbf, sbi, sbc, sbo;

    static Matrix sigmoid(const Matrix& x) {
        return x.apply([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
    }

    static Matrix tanh_m(const Matrix& x) {
        return x.apply([](double v) { return std::tanh(v); });
    }

    static Matrix sigmoid_deriv(const Matrix& s) {
        return s.apply([](double v) { return v * (1.0 - v); });
    }

    static Matrix tanh_deriv(const Matrix& t) {
        return t.apply([](double v) { return 1.0 - v * v; });
    }

public:
    LSTMCell() : input_dim(0), hidden_dim(0) {}

    LSTMCell(size_t in_dim, size_t h_dim)
        : input_dim(in_dim), hidden_dim(h_dim) {

        Wf = Matrix::xavier(in_dim, h_dim);
        Wi = Matrix::xavier(in_dim, h_dim);
        Wc = Matrix::xavier(in_dim, h_dim);
        Wo = Matrix::xavier(in_dim, h_dim);

        Uf = Matrix::xavier(h_dim, h_dim);
        Ui = Matrix::xavier(h_dim, h_dim);
        Uc = Matrix::xavier(h_dim, h_dim);
        Uo = Matrix::xavier(h_dim, h_dim);

        bf = Matrix::zeros(1, h_dim);
        bi = Matrix::zeros(1, h_dim);
        bc = Matrix::zeros(1, h_dim);
        bo = Matrix::zeros(1, h_dim);

        h_prev = Matrix::zeros(1, h_dim);
        c_prev = Matrix::zeros(1, h_dim);

        sWf.init(in_dim, h_dim); sWi.init(in_dim, h_dim);
        sWc.init(in_dim, h_dim); sWo.init(in_dim, h_dim);
        sUf.init(h_dim, h_dim); sUi.init(h_dim, h_dim);
        sUc.init(h_dim, h_dim); sUo.init(h_dim, h_dim);
        sbf.init(1, h_dim); sbi.init(1, h_dim);
        sbc.init(1, h_dim); sbo.init(1, h_dim);
    }

    void reset_state() {
        h_prev = Matrix::zeros(1, hidden_dim);
        c_prev = Matrix::zeros(1, hidden_dim);
    }

    Matrix forward(const Matrix& x) {
        x_cache = x;

        Matrix inp = x;
        if (x.cols == 1 && x.rows > 1) inp = x.T();

        ft = sigmoid(inp * Wf + h_prev * Uf + bf);
        it = sigmoid(inp * Wi + h_prev * Ui + bi);
        ct_bar = tanh_m(inp * Wc + h_prev * Uc + bc);
        ot = sigmoid(inp * Wo + h_prev * Uo + bo);

        c_t = ft.hadamard(c_prev) + it.hadamard(ct_bar);
        h_t = ot.hadamard(tanh_m(c_t));

        c_prev = c_t;
        h_prev = h_t;

        return h_t;
    }

    Matrix backward(const Matrix& dh_next) {
        Matrix tanh_c = tanh_m(c_t);

        Matrix dot = dh_next.hadamard(tanh_c).hadamard(sigmoid_deriv(ot));

        Matrix dc = dh_next.hadamard(ot).hadamard(tanh_deriv(tanh_c));

        Matrix dft = dc.hadamard(c_prev).hadamard(sigmoid_deriv(ft));
        Matrix dit = dc.hadamard(ct_bar).hadamard(sigmoid_deriv(it));
        Matrix dct_bar = dc.hadamard(it).hadamard(tanh_deriv(ct_bar));

        Matrix inp = x_cache;
        if (inp.cols == 1 && inp.rows > 1) inp = inp.T();

        dWf = inp.T() * dft; dWi = inp.T() * dit;
        dWc = inp.T() * dct_bar; dWo = inp.T() * dot;

        dUf = h_prev.T() * dft; dUi = h_prev.T() * dit;
        dUc = h_prev.T() * dct_bar; dUo = h_prev.T() * dot;

        dbf = dft; dbi = dit;
        dbc = dct_bar; dbo = dot;

        Matrix dx = dft * Wf.T() + dit * Wi.T() + dct_bar * Wc.T() + dot * Wo.T();

        return dx;
    }

    void update_weights(Optimizer& opt) {
        opt.update(Wf, dWf, sWf); opt.update(Wi, dWi, sWi);
        opt.update(Wc, dWc, sWc); opt.update(Wo, dWo, sWo);
        opt.update(Uf, dUf, sUf); opt.update(Ui, dUi, sUi);
        opt.update(Uc, dUc, sUc); opt.update(Uo, dUo, sUo);
        opt.update(bf, dbf, sbf); opt.update(bi, dbi, sbi);
        opt.update(bc, dbc, sbc); opt.update(bo, dbo, sbo);
    }

    Matrix get_hidden() const { return h_t; }
    Matrix get_cell() const { return c_t; }

    size_t get_input_dim() const { return input_dim; }
    size_t get_hidden_dim() const { return hidden_dim; }

    size_t param_count() const {
        return 4 * (input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim);
    }

    std::string info() const {
        return "LSTM(" + std::to_string(input_dim) + " -> " +
               std::to_string(hidden_dim) + ") [" +
               std::to_string(param_count()) + " params]";
    }
};

class LSTMNetwork {
    std::vector<LSTMCell> cells;
    DenseLayer output_layer;
    Optimizer opt;

public:
    LSTMNetwork() = default;

    LSTMNetwork(size_t input_dim, std::vector<size_t> hidden_dims,
                size_t output_dim, Activation out_act = Activation::Linear) {

        size_t prev = input_dim;
        for (size_t h : hidden_dims) {
            cells.emplace_back(prev, h);
            prev = h;
        }
        output_layer = DenseLayer(prev, output_dim, out_act);
    }

    void set_optimizer(Optimizer o) { opt = o; }

    Matrix forward(const Matrix& x) {
        Matrix h = x;
        for (auto& cell : cells)
            h = cell.forward(h);
        return output_layer.forward(h);
    }

    Matrix forward_sequence(const std::vector<Matrix>& seq) {
        reset();
        Matrix out;
        for (const auto& x : seq)
            out = forward(x);
        return out;
    }

    void backward(const Matrix& grad) {
        Matrix g = output_layer.backward(grad);
        for (int i = static_cast<int>(cells.size()) - 1; i >= 0; --i)
            g = cells[i].backward(g);
    }

    void update(Optimizer& optimizer) {
        for (auto& cell : cells)
            cell.update_weights(optimizer);
        output_layer.update_weights(optimizer);
    }

    void reset() {
        for (auto& cell : cells)
            cell.reset_state();
    }

    size_t total_params() const {
        size_t total = output_layer.param_count();
        for (const auto& cell : cells)
            total += cell.param_count();
        return total;
    }

    void summary() const {
        std::cout << util::color::BOLD << util::color::MAGENTA
                  << "\n  ┌──────────────────────────────────────────────┐\n"
                  << "  │  LSTM Network                                │\n"
                  << "  ├──────────────────────────────────────────────┤\n"
                  << util::color::RESET;
        for (size_t i = 0; i < cells.size(); ++i)
            std::cout << "  │  " << cells[i].info() << std::string(44 - cells[i].info().size(), ' ') << "│\n";
        std::cout << "  │  " << output_layer.info() << std::string(44 - output_layer.info().size(), ' ') << "│\n";
        std::cout << util::color::BOLD << util::color::YELLOW
                  << "  │  Total: " << total_params() << " params" << std::string(32 - std::to_string(total_params()).size(), ' ') << "│\n"
                  << util::color::MAGENTA
                  << "  └──────────────────────────────────────────────┘\n\n"
                  << util::color::RESET;
    }
};

} // namespace xasm
