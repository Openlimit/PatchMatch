#ifndef PATCHMATCH_PATCHMATCH_H
#define PATCHMATCH_PATCHMATCH_H

#include <eigen3/Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>

template<typename T, int _channel>
class ImageMat {
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

    ImageMat(int rows, int cols) {
        for (int i = 0; i < _channel; ++i) {
            m[i].resize(rows, cols);
        }
    }

    long rows() {
        return m[0].rows();
    }

    long cols() {
        return m[0].cols();
    }

    T &operator()(int i, int j, int k) {
        return m[k](i, j);
    }

    Mat &channel(int d) {
        return m[d];
    }

    float patch_dist(ImageMat *other, int i1, int j1, int i2, int j2, int p_h, int p_w) {
        float dist = 0;
        for (int k = 0; k < _channel; ++k) {
            Mat block1 = m[k].block(i1, j1, p_h, p_w);
            Mat block2 = other->channel(k).block(i2, j2, p_h, p_w);
            Mat block3 = block1 - block2;
            Mat block4 = block3.array().square();
            dist += block4.sum();
        }
        dist /= (p_h * p_w);
        return dist;
    }

private:
    Mat m[_channel];
};

class PatchMatch {
public:
    PatchMatch(ImageMat<int, 3> *src, ImageMat<int, 3> *dst,
               int patch_size, int step = 1,
               ImageMat<int, 2> *init_nnf = nullptr,
               ImageMat<int, 2> *src_gradient = nullptr,
               ImageMat<int, 2> *dst_gradient = nullptr,
               float g_alpha = 0.5, bool cal_gradient = false) {
        this->src = src;
        this->dst = dst;
        this->patch_size = patch_size;
        this->step = step;
        this->nnf = init_nnf;
        this->src_gradient = src_gradient;
        this->dst_gradient = dst_gradient;
        this->g_alpha = g_alpha;
        this->cal_gradient = cal_gradient;

        initialization();
    }

    ~PatchMatch() {
        if (is_init && nnf != nullptr)
            delete (nnf);
    }

    void NNS(int itr = 5) {
        for (int k = 0; k < itr; ++k) {
            std::cout << "iteration " << k << std::endl;
            if (k % 2 == 0) {
                for (int i = src_h - 1; i >= 0; i -= step) {
                    for (int j = src_w - 1; j >= 0; j -= step) {
                        propagation(i, j, false);
                        random_search(i, j);
                    }
                }
            } else {
                for (int i = 0; i < src_h; i += step) {
                    for (int j = 0; j < src_w; j += step) {
                        propagation(i, j, true);
                        random_search(i, j);
                    }
                }
            }
        }
    }

    void reconstruction(ImageMat<int, 3> *out) {
        for (int i = 0; i < src_h; ++i) {
            for (int j = 0; j < src_w; ++j) {
                int x_c = (*nnf)(i, j, 0);
                int y_c = (*nnf)(i, j, 1);
                for (int k = 0; k < 3; ++k) {
                    (*out)(i, j, k) = (*dst)(x_c, y_c, k);
                }
            }
        }
    }

    ImageMat<int, 3> *src;
    ImageMat<int, 3> *dst;
    ImageMat<int, 2> *nnf;
    Eigen::MatrixXf dist;
    ImageMat<int, 2> *src_gradient;
    ImageMat<int, 2> *dst_gradient;
    bool is_init;

private:
    void initialization() {
        src_h = src->rows();
        src_w = src->cols();
        dst_h = dst->rows();
        dst_w = dst->cols();

        if (nnf == nullptr) {
            nnf = new ImageMat<int, 2>(src_h, src_w);
            for (int i = 0; i < src_h; i++) {
                for (int j = 0; j < src_w; j++) {
                    (*nnf)(i, j, 0) = i;
                    (*nnf)(i, j, 1) = j;
                }
            }
            is_init = false;
        } else {
            is_init = true;
        }

        dist.resize(src_h, src_w);
        for (int i = 0; i < src_h; i++) {
            for (int j = 0; j < src_w; j++) {
                dist(i, j) = -1;
            }
        }
    }

    float cal_distance(int x, int y, int x_c, int y_c) {
        int p = patch_size / 2;
        int dx0 = std::min(std::min(x, x_c), p);
        int dx1 = std::min(std::min(src_h - x, dst_h - x_c), p + 1);
        int dy0 = std::min(std::min(y, y_c), p);
        int dy1 = std::min(std::min(src_w - y, dst_w - y_c), p + 1);
        int p_h = dx0 + dx1;
        int p_w = dy0 + dy1;
        int num = p_h * p_w;

        int pi1 = x - dx0;
        int pj1 = y - dy0;
        int pi2 = x_c - dx0;
        int pj2 = y_c - dy0;

        float d = src->patch_dist(dst, pi1, pj1, pi2, pj2, p_h, p_w);

        if (cal_gradient) {
            float d_g = src_gradient->patch_dist(dst_gradient, pi1, pj1, pi2, pj2, p_h, p_w);
            d = d * (1 - g_alpha) + d_g * g_alpha;
        }

        return d;
    }

    void update(int x, int y, int nx, int ny) {
        if (dist(x, y) < 0) {
            dist(x, y) = cal_distance(x, y, (*nnf)(x, y, 0), (*nnf)(x, y, 1));
        }

        int x_c = x + (*nnf)(nx, ny, 0) - nx;
        int y_c = y + (*nnf)(nx, ny, 1) - ny;
        if (x_c == (*nnf)(x, y, 0) && y_c == (*nnf)(x, y, 1))
            return;

        if (x_c >= 0 && x_c < dst_h && y_c >= 0 && y_c < dst_w) {
            float d = cal_distance(x, y, x_c, y_c);
            if (d < dist(x, y)) {
                dist(x, y) = d;
                (*nnf)(x, y, 0) = x_c;
                (*nnf)(x, y, 1) = y_c;
            }
        }
    }

    void propagation(int x, int y, bool is_odd) {
        int nx, ny;
        int x_c, y_c;
        float d;

        if (is_odd) {
            update(x, y, std::max(x - step, 0), y);
            update(x, y, x, std::max(y - step, 0));
        } else {
            update(x, y, std::min(x + step, src_h - 1), y);
            update(x, y, x, std::min(y + step, src_w - 1));
        }
    }

    void random_search(int x, int y, float alpha = 0.5) {
        int search_h = dst_h * std::pow(alpha, 4);
        int search_w = dst_w * std::pow(alpha, 4);
        int x_c = (*nnf)(x, y, 0);
        int y_c = (*nnf)(x, y, 1);

        std::default_random_engine generator;
        while (search_h > 0 && search_w > 0) {
            int search_min_x = std::max(x_c - search_h, 0);
            int search_max_x = std::min(x_c + search_h, dst_h - 1);
            std::uniform_int_distribution<int> distribution1(search_min_x, search_max_x);
            int random_x_c = distribution1(generator);

            int search_min_y = std::max(y_c - search_w, 0);
            int search_max_y = std::min(y_c + search_w, dst_w - 1);
            std::uniform_int_distribution<int> distribution2(search_min_y, search_max_y);
            int random_y_c = distribution2(generator);

            float d = cal_distance(x, y, random_x_c, random_y_c);
            if (d < dist(x, y)) {
                dist(x, y) = d;
                (*nnf)(x, y, 0) = random_x_c;
                (*nnf)(x, y, 1) = random_y_c;
            }
            search_h *= alpha;
            search_w *= alpha;
        }
    }


    int patch_size;
    int step;
    float g_alpha;
    bool cal_gradient;
    int src_h;
    int src_w;
    int dst_h;
    int dst_w;
};


#endif //PATCHMATCH_PATCHMATCH_H
