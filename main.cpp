#include <cstddef>
#include <iostream>
#include <vector>
#include <cmath>    // std::floor, std::pow, std::sqrt のため
#include <random>
#include <string>
#include <fstream>  // ファイル出力のため
#include <iomanip>  // std::fixed, std::setprecision のため

#include <Eigen/Dense>  // Eigen のベクトルと行列のため

using Vec2 = Eigen::Vector2d;
using Mat2 = Eigen::Matrix2d;
using Vec2i = Eigen::Vector2i;

// 定数
const int n_particles = 8192;   // 粒子数
const int n_grid = 128;         // グリッド解像度(N x N グリッド)
const double dx = 1.0 / n_grid; // グリッドセルのサイズ
const double dt = 2e-4;         // 時間ステップ

const double p_rho = 1.0;                       // 粒子の密度
const double p_vol = std::pow(dx * 0.5, 2.0);   // 粒子の体積（2Dなので面積）
const double p_mass = p_vol * p_rho;            // 粒子の質量
const double gravity = 9.81;                    // 重力加速度
const int bound = 3;                            // 境界条件の厚さ（グリッドセル単位）
const double E = 400.0;                         // ヤング率（弾性）

// 粒子データ
std::vector<Vec2> particle_x(n_particles);                          // 位置
std::vector<Vec2> particle_v(n_particles);                          // 速度
std::vector<Mat2> particle_C(n_particles, Mat2::Zero());    // アフィン運動量行列（APIC）
std::vector<double> particle_J(n_particles, 1.0);              // 変形勾配の行列式（体積変化率）

// グリッドデータ
// グリッド用のフラット化された1Dベクトル: (ix, iy) は iy * n_grid + ix でアクセス
std::vector<Vec2> grid_v(n_grid * n_grid, Vec2::Zero());    // グリッドノードの速度
std::vector<double> grid_m(n_grid * n_grid, 0.0);              // グリッドノードの質量

// 1Dのgrid_vおよびgrid_mベクトルへの2Dグリッドインデックスのためのヘルパー関数
inline int grid_idx(int ix, int iy) {
    // 安全のための境界チェック（ただし、内部ループが範囲外アクセスを防ぐはず）
    if (ix < 0 || ix >= n_grid || iy < 0 || iy >= n_grid) {
        // ロジックが正しければ、これは起こらない
        return -1;
    }
    return iy * n_grid + ix;    // 行優先順序（iyが行、ixが列）
}

inline int grid_idx(const Vec2i& cell_coord) {
    return grid_idx(cell_coord.x(), cell_coord.y());
}

// substep関数: コアMPMシミュレーションステップ
void substep() {
    // 1. グリッドの初期化（質量と速度をクリア）
    std::fill(grid_v.begin(), grid_v.end(), Vec2::Zero());
    std::fill(grid_m.begin(), grid_m.end(), 0.0);

    // 2. P2G（粒子からグリッドへの転送）
    for (int p = 0; p < n_particles; ++p) {
        Vec2 Xp_grid_coords = particle_x[p] / dx;   // グリッド空間単位での粒子位置

        // ベースグリッドセルインデックス（3x3ステンシルの中央セルの左下ノード）
        Vec2i base_coord = (Xp_grid_coords.array() - 0.5).floor().template cast<int>();

        Vec2 fx = Xp_grid_coords - base_coord.template cast<double>();  // base_coordに対する相対的な小数位置

        // 各次元の2次Bスプライン補間ウェイト
        std::array<double, 3> w_fx_terms, w_fy_terms;

        w_fx_terms[0] = 0.5 * std::pow(1.5 - fx.x(), 2.0);
        w_fx_terms[1] = 0.75 - std::pow(fx.x() - 1.0, 2.0);
        w_fx_terms[2] = 0.5 * std::pow(fx.x() - 0.5, 2.0);

        w_fy_terms[0] = 0.5 * std::pow(1.5 - fx.y(), 2.0);
        w_fy_terms[1] = 0.75 - std::pow(fx.y() - 1.0, 2.0);
        w_fy_terms[2] = 0.5 * std::pow(fx.y() - 0.5, 2.0);

        // 応力項（J、変形勾配の行列式から）
        // MLS-MPMの場合、これは弾性モデルに関連（例: Jに基づくNeo-Hookean）
        // stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        // これは等方的な圧力のような力のためのスカラ応力
        double stress_scalar = -dt * 4.0 * E * p_vol * (particle_J[p] - 1.0) / (dx * dx);
        Mat2 stress_tensor = Mat2::Identity() * stress_scalar;  // 等方性応力

        Mat2 affine_term = stress_tensor + p_mass * particle_C[p];

        // 3x3グリッド近傍を反復処理
        for (int i_offset = 0; i_offset < 3; ++i_offset) {
            for (int j_offset = 0; j_offset < 3; ++j_offset) {
                Vec2i offset_vec(i_offset, j_offset);
                Vec2i current_grid_node_coord = base_coord + offset_vec;

                // グリッドノードが境界内にあることを確認
                if (current_grid_node_coord.x() < 0 || current_grid_node_coord.x() >= n_grid ||
                    current_grid_node_coord.y() < 0 || current_grid_node_coord.y() >= n_grid ) {
                    continue;
                    }

                Vec2 dpos = (offset_vec.template cast<double>() - fx) * dx; // 粒子からグリッドノードまでの距離（ステンシル相対）
                double weight = w_fx_terms[i_offset] * w_fy_terms[j_offset];    // 結合された2D補間ウェイト

                int g_linear_idx = grid_idx(current_grid_node_coord);
                if (g_linear_idx == -1) continue;  // 上記のチェックで発生しないはず

                grid_v[g_linear_idx] += weight * (p_mass * particle_v[p] + affine_term * dpos);
                grid_m[g_linear_idx] += weight * p_mass;
            }
        }
    }

    // 3. グリッド操作（速度の正規化、重力の適用、境界条件
    for (int iy = 0; iy < n_grid; ++iy) {   // iy はグリッドのy座標に対応
        for (int ix = 0; ix < n_grid; ++ix) {   // ix はグリッドのx座標に対応
            int g_linear_idx = grid_idx(ix, iy);
            if (g_linear_idx == -1) continue;

            if (grid_m[g_linear_idx] > 1e-15) { // ゼロ除算を避けるために非常に小さい質量をチェック
                grid_v[g_linear_idx] /= grid_m[g_linear_idx];   // 速度を正規化
            }
            grid_v[g_linear_idx].y() -= dt * gravity;   // 重力を適用

            // 境界条件（固定境界）
            // ノードが境界付近にあり、速度が外向きの場合、その成分を0にする。
            if (ix < bound && grid_v[g_linear_idx].x() < 0) grid_v[g_linear_idx].x() = 0;
            if (ix >= n_grid - bound && grid_v[g_linear_idx].x() > 0) grid_v[g_linear_idx].x() = 0; // 上限アクセスの修正
            if (iy < bound && grid_v[g_linear_idx].y() < 0) grid_v[g_linear_idx].y() = 0;
            if (iy >= n_grid - bound && grid_v[g_linear_idx].y() > 0) grid_v[g_linear_idx].y() = 0; // 上限アクセスの修正
        }
    }

    // 4. G2P（グリッドから粒子への転送）
    for (int p = 0; p < n_particles; ++p) {
        Vec2 Xp_grid_coords = particle_x[p] / dx;
        Vec2i base_coord = (Xp_grid_coords.array() - 0.5).floor().template cast<int>();
        Vec2 fx = Xp_grid_coords - base_coord.template cast<double>();

        std::array<double, 3> w_fx_terms, w_fy_terms;
        w_fx_terms[0] = 0.5 * std::pow(1.5 - fx.x(), 2.0);
        w_fx_terms[1] = 0.75 - std::pow(fx.x() - 1.0, 2.0);
        w_fx_terms[2] = 0.5 * std::pow(fx.x() - 0.5, 2.0);

        w_fy_terms[0] = 0.5 * std::pow(1.5 - fx.y(), 2.0);
        w_fy_terms[1] = 0.75 - std::pow(fx.y() - 1.0, 2.0);
        w_fy_terms[2] = 0.5 * std::pow(fx.y() - 0.5, 2.0);

        Vec2 new_v_p = Vec2::Zero();
        Mat2 new_C_p = Mat2::Zero();

        for (int i_offset = 0; i_offset < 3; ++i_offset) {
            for (int j_offset = 0; j_offset < 3; ++j_offset) {
                Vec2i offset_vec(i_offset, j_offset);
                Vec2i current_grid_node_coord = base_coord + offset_vec;

                if (current_grid_node_coord.x() < 0 || current_grid_node_coord.x() >= n_grid ||
                    current_grid_node_coord.y() < 0 || current_grid_node_coord.y() >= n_grid) {
                    continue;
                    }

                Vec2 dpos = (offset_vec.template cast<double>() - fx) * dx;
                double weight = w_fx_terms[i_offset] * w_fy_terms[j_offset];

                int g_linear_idx = grid_idx(current_grid_node_coord);
                if (g_linear_idx == -1) continue;
                Vec2 g_v_node = grid_v[g_linear_idx];

                new_v_p += weight * g_v_node;
                // Taichi: new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
                // Eigen: a と b の外積は a * b.transpose()
                new_C_p += 4.0 * weight * (g_v_node * dpos.transpose()) / (dx * dx);
            }
        }

        particle_v[p] = new_v_p;
        particle_x[p] += dt * particle_v[p];    // 粒子を移流

        // J（変形勾配の行列式）を更新
        // J[p] *= 1 + dt * new_C.trace()
        particle_J[p] *= (1.0 + dt * new_C_p.trace());
        particle_C[p] = new_C_p;    // APICアフィン行列Cを更新
    }
}

// init_paritcles関数: 粒子のプロパティを初期化
void init_particles() {
    std::mt19937 rng(std::random_device{}());   // メルセンヌ・ツイスタ RNG
    std::uniform_real_distribution<double> dist(0.0, 1.0);  // [0, 1) のための分布

    for (int i = 0; i < n_particles; ++i) {
        // Taichi: x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        // [0,1]x[0,1]ドメイン内の[0.2, 0.6] x [0.2, 0.6]の正方形領域に粒子を配置
        particle_x[i] = Vec2(dist(rng) * 0.4 + 0.2, dist(rng) * 0.4 + 0.2);
        particle_v[i] = Vec2(0.0, -1.0);    // 初期の下向き速度
        particle_J[i] = 1.0;    // 初期J（変形なし）
        particle_C[i] = Mat2::Zero();   // 初期C行列
    }
}

// メインシミュレーターループ
int main() {
    std::cout << "Simple MPM Cpp シミュレーションを開始します..." << std::endl;
    std::cout << std::fixed << std::setprecision(5);    // doubleの出力精度を設定

    init_particles();

    const int num_frames_to_simulate = 200; // シミュレートする「フレーム」数
    const int substeps_per_frame = 50;  // Taichiの "for s in range(50): substep()" に対応

    for (int frame = 0; frame < num_frames_to_simulate; ++frame) {
        for (int s = 0; s < substeps_per_frame; ++ s) {
            substep();
        }

        // オプション: 数フレームごとに幾つかの粒子位置や統計情報を表示
        if (frame % 1 == 0) {
            // 1フレームごとに
            double total_kinetic_energy = 0.0;
            for (int p = 0; p < n_particles; ++p){
                total_kinetic_energy += 0.5 * p_mass * particle_v[p].squaredNorm();
            }
            std::cout << "フレーム " << frame << ": 粒子0 位置: ("
                      << particle_x[0].x() << ", " << particle_x[0].y() << "), "
                      << "総運動エネルギー: " << total_kinetic_energy <<  std::endl;
        }
    }

    std::cout << "シミュレーションが終了しました。" << std::endl;

    return 0;
}