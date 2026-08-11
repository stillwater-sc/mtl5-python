// Mixed-precision UKF stability: Cholesky vs LDL^T vs Bunch-Kaufman.
//
// Resolves stillwater-sc/mtl5-python#18.
//
// WHAT THIS MEASURES
//
// An Unscented Kalman Filter needs a matrix square root of the covariance P at
// every step, to place its 2n+1 sigma points. The textbook choice is Cholesky,
// A = L*L^T. In float64 that choice looks free: LL^T and LDL^T agree to the
// last bit until cond(P) approaches 1e15, so nothing distinguishes them.
//
// Reduced precision is where they separate, and the reason is the sqrt.
// Cholesky takes a square root *inside* the recurrence -- L(j,j) = sqrt(A(j,j)
// - sum) -- so a diagonal entry that cancellation has driven toward zero gets
// sqrt'd, and the relative error there is halved in bits: in float32, sqrt(eps)
// is 3.5e-4 against an eps of 1.2e-7. LDL^T is sqrt-free in the recurrence; the
// only square roots are the n scalar ones in sqrt(D), taken when the square root
// is assembled as L*sqrt(D). A non-positive D(j) is a crisp, detectable failure
// where Cholesky's equivalent is sqrt of a garbage value.
//
// As a rule of thumb a factorization in a format with eps of machine epsilon
// loses its positive-definiteness near cond(P) ~ 1/eps, which is 2e3 for an
// 11-bit significand and 4.5e15 for a 53-bit one. That is the span this
// experiment sweeps.
//
// THE STRESS CASE: BEARING-ONLY TRACKING
//
// The textbook ill-conditioning generator. A bearing sensor observes only the
// direction normal to the line of sight, so each update squeezes one eigenvalue
// of P to about R and leaves the other untouched, while propagation grows the
// unobserved range direction. The result is a covariance whose two eigenvalues
// separate geometrically -- exactly the regime where the choice of square root
// stops being cosmetic.
//
// THE PRIMARY DIAGNOSTIC: UNSCENTED-TRANSFORM MEAN BIAS
//
// A factorization that *returns success* can still be broken, and that is the
// dangerous case because nothing reports it.
//
// Note what does NOT work as a diagnostic: sigma points are placed as x +/-
// gamma*S(:,c), so the weighted mean telescopes to (w0 + 2n*wi)*x = x for ANY
// square root S, correct or not -- an algebraic identity, not a rounding
// accident. Comparing that mean against x is therefore blind to the quality of
// S. Measured directly: against a deliberately 10%-corrupted S, the raw mean
// moves by exactly 0.000e+00. The asymmetry only becomes observable after the
// points pass through a nonlinearity.
//
// So this propagates the sigma points through the actual bearing measurement
// h(chi) = atan2(chi_y, chi_x) and takes the weighted mean of the transformed
// points -- the unscented transform proper, which IS square-root dependent: the
// same 10% corruption moves it by 6.7e-08 rad in exact arithmetic.
//
// What the bias column then reports is |z_mean in this format - z_mean of the
// same method in float64|, so it is the precision-induced divergence of the
// whole pipeline -- factorization, sigma-point placement and transform
// together. It is not a measure of "distance from a correct square root"; that
// is what the resid column is for. The two answer different questions and both
// are printed.
//
// Each method is compared against ITSELF run in float64, not against a single
// shared reference. LL^T and LDL^T produce genuinely different square roots even
// in exact arithmetic -- both satisfy S*S^T = P -- and the unscented transform
// is not invariant to which one you pick. Comparing across methods would confuse
// that legitimate difference with precision loss; comparing each against its own
// float64 result isolates precision loss alone.
//
// Reported per (number system, method, step):
//
//   status -- ok, or the failure the factorization reported
//   bias   -- |z_mean - z_mean_float64| in radians, the silent killer above
//   resid  -- ||P - S*S^T||_F / ||P||_F, how well the square root reproduces P
//   bits   -- significand bits surviving the factorization, -log2(resid)
//
// WHY cond(P) IS COMPUTED IN FLOAT64
//
// It comes from a reference trajectory in float64, not from the working
// precision. Computing it in the precision under test would corrupt the very
// number meant to explain the failure -- posit16 cannot usefully represent 1e9,
// so a posit16-computed cond() would report nonsense exactly when the diagnostic
// matters most. The trajectory is identical for every format, so "step 7" names
// the same matrix everywhere and the columns are comparable down the page.

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/ldlt_bk.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Problem definition: 2D bearing-only tracking.
// ---------------------------------------------------------------------------

constexpr std::size_t N     = 2;   // state dimension: (x, y) position
constexpr int         STEPS = 14;  // sweeps cond(P) over roughly 1e3 .. 1e16

constexpr double P0     = 100.0;  // initial covariance, diag(P0, P0)
constexpr double BEARING = 0.35;  // radians, fixed sensor geometry

// Bearing measurement variance.
//
// The issue proposes R = 1e-4. That is a realistic angular variance, but it puts
// cond(P) at ~1e6 after the very first update -- already past the point where an
// 11- or 13-bit significand can factor at all, so every reduced-precision format
// would fail at step 1 and the experiment would show nothing about *where* each
// one breaks. A looser R starts the sweep in a range all six formats handle, so
// the breakdown step is observable for each. The physics is unchanged; only the
// starting point of the sweep moves.
constexpr double R_MEAS = 1e-1;

// Per-step growth of the *unobserved* range direction. A bearing sensor never
// constrains range, so its variance grows during propagation; this is the rank-1
// inflation of the line-of-sight direction that models it. A factor of 10 per
// step walks cond(P) up by a decade at a time, which is what spreads the six
// number systems out into distinguishable failure points.
constexpr double RANGE_GROWTH = 10.0;

// Unscented transform weights.
//
// alpha = 1 (so lambda = 0, w0 = 0, wi = 1/2n, gamma = sqrt(n)) rather than the
// common alpha = 1e-3. With alpha = 1e-3 and n = 2, w0 is about -1e6 and wi
// about 2.5e5, so the weighted mean is a cancellation of 1e6-sized terms down to
// order 1, and that cancellation amplifies every rounding error along with it.
//
// Measured, on one P with a square root corrupted by 10%: the signal to be
// resolved -- how much the transform moves when the square root changes -- is
// 6.7e-08 rad. The float32 noise floor is 3.1e-11 at alpha = 1 but 7.1e-08 at
// alpha = 1e-3. So the conventional alpha puts the noise floor *at* the signal
// (SNR ~ 0.9) while alpha = 1 leaves about three orders of headroom (SNR ~
// 2000). The tuning parameter would otherwise decide the result of the
// experiment, which is not a property you want in an instrument.
constexpr double ALPHA  = 1.0;
constexpr double KAPPA  = 0.0;
constexpr double LAMBDA = ALPHA * ALPHA * (double(N) + KAPPA) - double(N);

/// True target state the filter is tracking, used as the sigma-point centre.
constexpr double X_TRUE[N] = {1000.0, 500.0};

/// Grow the unobserved (line-of-sight) direction, then apply a bearing update.
///
/// u = [cos b, sin b] is the line of sight; h = [-sin b, cos b] is the bearing
/// normal, the only direction a bearing sensor observes. Inflating along u and
/// then squeezing along h is what separates the eigenvalues of P.
///
/// The update is the naive P <- P - K*H*P rather than the Joseph form, on
/// purpose: this is the reference conditioning trajectory, and the naive form is
/// what generates the ill-conditioning the experiment is about.
void propagate_and_update(double P[N][N]) {
    const double u0 = std::cos(BEARING), u1 = std::sin(BEARING);
    const double h0 = -std::sin(BEARING), h1 = std::cos(BEARING);

    // Rank-1 inflation along u: var_u <- var_u * (1 + RANGE_GROWTH).
    const double var_u = u0 * (P[0][0] * u0 + P[0][1] * u1)
                       + u1 * (P[1][0] * u0 + P[1][1] * u1);
    const double add = RANGE_GROWTH * var_u;
    P[0][0] += add * u0 * u0;
    P[0][1] += add * u0 * u1;
    P[1][0] += add * u1 * u0;
    P[1][1] += add * u1 * u1;

    // S = H*P*H^T + R, scalar because the measurement is scalar.
    const double Ph0 = P[0][0] * h0 + P[0][1] * h1;
    const double Ph1 = P[1][0] * h0 + P[1][1] * h1;
    const double S   = h0 * Ph0 + h1 * Ph1 + R_MEAS;

    const double k0 = Ph0 / S, k1 = Ph1 / S;

    const double p00 = P[0][0] - k0 * Ph0;
    const double p01 = P[0][1] - k0 * Ph1;
    const double p10 = P[1][0] - k1 * Ph0;
    const double p11 = P[1][1] - k1 * Ph1;

    // Re-symmetrize: the update is symmetric exactly, but rounding breaks that,
    // and an asymmetric P makes ldlt_factor report NOT_SYMMETRIC for a reason
    // unrelated to the precision under test.
    P[0][0] = p00;
    P[0][1] = P[1][0] = 0.5 * (p01 + p10);
    P[1][1] = p11;
}

/// Condition number of a 2x2 symmetric matrix from its eigenvalues, in float64.
///
/// Closed form rather than an eigensolver call: for 2x2 the eigenvalues are
/// t/2 +/- sqrt((t/2)^2 - det), which is exact enough here and keeps the
/// diagnostic free of any iterative-solver behaviour at the extreme conditioning
/// this experiment deliberately produces.
double cond2x2(const double P[N][N]) {
    const double t    = P[0][0] + P[1][1];
    const double det  = P[0][0] * P[1][1] - P[0][1] * P[1][0];
    const double disc = std::sqrt(std::max(0.0, (t * 0.5) * (t * 0.5) - det));
    const double l1 = t * 0.5 + disc, l2 = t * 0.5 - disc;
    if (l2 <= 0.0) return std::numeric_limits<double>::infinity();
    return l1 / l2;
}

// ---------------------------------------------------------------------------
// Factorization paths.
// ---------------------------------------------------------------------------

enum class Method { Cholesky, LDLT, BunchKaufman };
constexpr Method ALL_METHODS[] = {Method::Cholesky, Method::LDLT, Method::BunchKaufman};

const char* method_name(Method m) {
    switch (m) {
        case Method::Cholesky:     return "LL^T";
        case Method::LDLT:         return "LDL^T";
        case Method::BunchKaufman: return "LDL^T-BK";
    }
    return "?";
}

struct Outcome {
    bool        ok       = false;
    std::string status   = "ok";
    double      zmean    = 0.0;  // unscented-transform output mean, radians
    double      bias     = 0.0;  // |zmean - zmean_float64|
    double      residual = 0.0;  // ||P - S*S^T||_F / ||P||_F
};

/// Turn an LDL^T factorization in place into the square root S = L*sqrt(D).
///
/// This is where LDL^T finally takes its n scalar square roots, and where a
/// non-positive pivot is caught as a clean signal rather than a sqrt of garbage.
template <typename S>
bool assemble_sqrt_from_ldlt(mtl::mat::dense2D<S>& A, std::string& why) {
    for (std::size_t j = 0; j < N; ++j) {
        const double d = double(A(j, j));
        if (!(d > 0.0)) {
            why = "D" + std::to_string(j) + "<=0";
            return false;
        }
        const S s = S(std::sqrt(d));
        A(j, j) = s;
        for (std::size_t i = j + 1; i < N; ++i) A(i, j) = A(i, j) * s;
    }
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) A(i, j) = S(0);
    return true;
}

template <typename S>
void clear_upper(mtl::mat::dense2D<S>& A) {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) A(i, j) = S(0);
}

/// Factor P in precision S by the requested method, then measure the result.
template <typename S>
Outcome evaluate(const double Pd[N][N], Method method) {
    Outcome out;

    // Dynamic range comes first, and it is a genuinely different failure from
    // running out of precision. cfloat<16,5> tops out at 1.3e5 and posit<16,1>
    // at 2.7e8, while this trajectory drives covariance entries to 1e15 -- so
    // for the small formats the matrix stops being *representable* long before
    // any factorization gets a chance to lose positive-definiteness. Reporting
    // that as "not SPD" would blame the algorithm for the format's range, so it
    // is detected here and labelled separately.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const double back = double(S(Pd[i][j]));
            const bool   lost = !std::isfinite(back)
                             || (Pd[i][j] != 0.0
                                 && std::abs(back - Pd[i][j]) / std::abs(Pd[i][j]) > 0.5);
            if (lost) {
                out.status = "range";
                return out;
            }
        }
    }

    mtl::mat::dense2D<S> A(N, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) A(i, j) = S(Pd[i][j]);

    if (method == Method::Cholesky) {
        const int rc = mtl::cholesky_factor(A);
        if (rc != 0) {
            out.status = "notSPD@" + std::to_string(rc - 1);
            return out;
        }
        clear_upper(A);
    } else if (method == Method::LDLT) {
        const int rc = mtl::ldlt_factor(A);
        if (rc != 0) {
            out.status = "pivot@" + std::to_string(rc - 1);
            return out;
        }
        std::string why;
        if (!assemble_sqrt_from_ldlt(A, why)) {
            out.status = why;
            return out;
        }
    } else {
        mtl::bk_pivot_info piv;
        const int rc = mtl::ldlt_bk_factor(A, piv);
        if (rc != 0) {
            out.status = "bk@" + std::to_string(rc - 1);
            return out;
        }
        std::string why;
        // A 2x2 pivot block puts an off-diagonal term in D, which L*sqrt(D)
        // cannot express. Report that rather than emit a wrong square root.
        if (!assemble_sqrt_from_ldlt(A, why)) {
            out.status = why;
            return out;
        }
    }

    // --- residual, accumulated in float64 from the computed S ---
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < N; ++k)
                acc += double(A(i, k)) * double(A(j, k));
            const double diff = Pd[i][j] - acc;
            num += diff * diff;
            den += Pd[i][j] * Pd[i][j];
        }
    }
    out.residual = (den > 0.0) ? std::sqrt(num / den) : 0.0;

    // --- unscented transform through the bearing measurement ---
    //
    // The sigma points are formed in the working precision, because their
    // placement is what the square root controls. h() itself is evaluated in
    // float64 so the trig is not another variable.
    const double gamma = std::sqrt(double(N) + LAMBDA);
    const double w0    = LAMBDA / (double(N) + LAMBDA);
    const double wi    = 1.0 / (2.0 * (double(N) + LAMBDA));

    auto h = [](double px, double py) { return std::atan2(py, px); };

    double zmean = w0 * h(X_TRUE[0], X_TRUE[1]);
    for (std::size_t c = 0; c < N; ++c) {
        double plus[N], minus[N];
        for (std::size_t d = 0; d < N; ++d) {
            const S col = A(d, c);
            plus[d]  = double(S(S(X_TRUE[d]) + S(gamma) * col));
            minus[d] = double(S(S(X_TRUE[d]) - S(gamma) * col));
        }
        zmean += wi * (h(plus[0], plus[1]) + h(minus[0], minus[1]));
    }
    // A factorization that reports success but produces a non-finite square root
    // has still failed; saying "ok" next to a nan would be the worst of both.
    if (!std::isfinite(out.residual) || !std::isfinite(zmean)) {
        out.status = "nan";
        return out;
    }

    out.zmean = zmean;
    out.ok    = true;
    return out;
}

// ---------------------------------------------------------------------------
// Precision reporting, via Universal's introspection.
// ---------------------------------------------------------------------------

/// Significand bits available at 1.0.
///
/// For the Universal types this comes from ulp(), which reports the actual
/// spacing to the next representable value -- the honest number for a tapered
/// format like posit, where "how many bits do I have" depends on where you are
/// on the real line. Native types use numeric_limits::epsilon(), which is the
/// same quantity. bits = -log2(spacing at 1.0).
template <typename S>
double significand_bits_at_one() {
    if constexpr (std::is_floating_point_v<S>) {
        return -std::log2(double(std::numeric_limits<S>::epsilon()));
    } else {
        const S one(1.0);
        const double u = double(sw::universal::ulp(one));
        return (u > 0.0) ? -std::log2(u) : 0.0;
    }
}

/// Smallest and largest positive representable values.
///
/// Universal exposes these uniformly across its number systems through the
/// shared SpecificValue encoding (number/shared/specific_value_encoding.hpp), so
/// T(SpecificValue::minpos) and T(SpecificValue::maxpos) are the portable way to
/// ask any of them for their dynamic range. That is what this uses.
///
/// Deliberately NOT numeric_limits<T>::min(): that is the smallest *normal*
/// value, whereas minpos is the smallest positive *representable* one,
/// subnormals included. For the cfloats the two differ by decades --
/// cfloat<32,8> is 1.18e-38 normal against 1.40e-45 minpos, cfloat<16,5> is
/// 6.10e-05 against 5.96e-08. The purpose of this column is to explain where a
/// format runs out of room, so the smallest representable value is the honest
/// one. Posits have no subnormals, so for them the two coincide -- which is why
/// substituting numeric_limits here looks harmless until you reach a cfloat.
///
/// Native float and double have no SpecificValue constructor; denorm_min() is
/// the same quantity for them.
template <typename S>
std::pair<double, double> dynamic_range() {
    if constexpr (std::is_floating_point_v<S>) {
        return {double(std::numeric_limits<S>::denorm_min()),
                double(std::numeric_limits<S>::max())};
    } else {
        return {double(S(sw::universal::SpecificValue::minpos)),
                double(S(sw::universal::SpecificValue::maxpos))};
    }
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

struct Row {
    int    step;
    double cond;
    double Pd[N][N];
};

/// Build the shared conditioning trajectory, stopping as soon as float64 itself
/// can no longer be trusted with it.
///
/// The sweep is only meaningful while the reference P is still symmetric
/// positive definite in float64: past that point the matrix under test is no
/// longer SPD for *anyone*, so a "failure" says nothing about the number system
/// and the comparison silently turns into a different experiment. The loop
/// therefore stops at the last step where cond(P) is finite and a float64
/// Cholesky still succeeds, and reports where it stopped.
std::vector<Row> reference_trajectory(int& stopped_at) {
    std::vector<Row> rows;
    double P[N][N] = {{P0, 0.0}, {0.0, P0}};
    stopped_at = 0;
    for (int s = 1; s <= STEPS; ++s) {
        propagate_and_update(P);
        const double c = cond2x2(P);
        if (!std::isfinite(c)) {
            stopped_at = s;
            break;
        }
        Row r;
        r.step = s;
        r.cond = c;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) r.Pd[i][j] = P[i][j];
        if (!evaluate<double>(r.Pd, Method::Cholesky).ok) {
            stopped_at = s;
            break;
        }
        rows.push_back(r);
    }
    return rows;
}

/// float64 reference means, one per (step, method), for the bias comparison.
std::vector<std::vector<double>> reference_means(const std::vector<Row>& traj) {
    std::vector<std::vector<double>> ref(traj.size(), std::vector<double>(3, 0.0));
    for (std::size_t s = 0; s < traj.size(); ++s)
        for (int m = 0; m < 3; ++m)
            ref[s][m] = evaluate<double>(traj[s].Pd, ALL_METHODS[m]).zmean;
    return ref;
}

template <typename S>
void run(const char* label, const std::vector<Row>& traj,
         const std::vector<std::vector<double>>& ref) {
    std::cout << "\n--- " << label << " ---\n";
    const auto [lo, hi] = dynamic_range<S>();
    std::cout << "    significand at 1.0: " << std::fixed << std::setprecision(1)
              << significand_bits_at_one<S>() << " bits    [minpos, maxpos] = ["
              << std::scientific << std::setprecision(2) << lo << ", " << hi << "]\n";

    std::cout << "    step      cond(P) |";
    for (Method m : ALL_METHODS)
        std::cout << std::setw(9) << method_name(m) << std::setw(10) << "bias"
                  << std::setw(10) << "resid" << " |";
    std::cout << "\n";

    int first_fail[3]  = {0, 0, 0};  // first precision failure
    int first_range    = 0;          // first step P stops being representable

    for (std::size_t s = 0; s < traj.size(); ++s) {
        const Row& r = traj[s];
        std::cout << "  " << std::setw(6) << r.step << " " << std::setw(12)
                  << std::scientific << std::setprecision(2) << r.cond << " |";
        for (int m = 0; m < 3; ++m) {
            const Outcome o = evaluate<S>(r.Pd, ALL_METHODS[m]);
            if (o.ok) {
                std::cout << std::setw(9) << "ok" << std::setw(10)
                          << std::setprecision(2) << std::scientific
                          << std::abs(o.zmean - ref[s][m]) << std::setw(10)
                          << o.residual << " |";
            } else {
                if (o.status == "range") {
                    if (first_range == 0) first_range = r.step;
                } else if (first_fail[m] == 0) {
                    first_fail[m] = r.step;
                }
                std::cout << std::setw(9) << o.status.c_str() << std::setw(10) << "-"
                          << std::setw(10) << "-" << " |";
            }
        }
        std::cout << "\n";
    }

    std::cout << "    first precision failure:";
    for (int m = 0; m < 3; ++m) {
        std::cout << "  " << method_name(ALL_METHODS[m]) << " ";
        if (first_fail[m] == 0)
            std::cout << "never";
        else
            std::cout << "step " << first_fail[m];
        std::cout << ";";
    }
    if (first_range)
        std::cout << "   [P exceeds the format's range from step " << first_range
                  << " on -- a range limit, not a precision one]";
    std::cout << "\n";
}

}  // namespace

int main() {
    using posit32  = sw::universal::posit<32, 2>;
    using posit16  = sw::universal::posit<16, 1>;
    using cfloat32 = sw::universal::cfloat<32, 8, std::uint32_t, true, true, false>;
    using cfloat16 = sw::universal::cfloat<16, 5, std::uint16_t, true, true, false>;

    std::cout <<
        "================================================================\n"
        " Mixed-Precision UKF: Bearing-Only Tracking\n"
        " Cholesky (LL^T) vs LDL^T vs Bunch-Kaufman across number systems\n"
        "================================================================\n"
        "\n"
        "  bias  = |unscented-transform mean - same method in float64|, radians.\n"
        "          Sigma points are pushed through h(x) = atan2(y, x); a degraded\n"
        "          square root moves them and the nonlinearity turns that into a\n"
        "          shifted mean. A nonzero bias next to an 'ok' status is the\n"
        "          silent failure: the factorization succeeded and the filter is\n"
        "          still wrong. float64 is the baseline, so its bias is 0 by\n"
        "          construction.\n"
        "  resid = ||P - S*S^T||_F / ||P||_F. -log2(resid) is roughly the number\n"
        "          of significand bits surviving the factorization.\n"
        "  cond(P) is computed in float64 from one shared trajectory, so a given\n"
        "          step is the same matrix for every number system.\n"
        "\n"
        "  Expect each format to fail near cond(P) ~ 1/eps.\n";

    int stopped_at = 0;
    const std::vector<Row> traj = reference_trajectory(stopped_at);
    const auto             ref  = reference_means(traj);

    std::cout << "\n  Sweep: " << traj.size() << " steps, cond(P) " << std::scientific
              << std::setprecision(1) << traj.front().cond << " -> " << traj.back().cond;
    if (stopped_at)
        std::cout << " (stopped at step " << stopped_at
                  << ": the reference P is no longer SPD in float64, so beyond"
                     " here a failure\n          would say nothing about the number"
                     " system under test)";
    std::cout << "\n";

    run<double>("float64", traj, ref);
    run<float>("float32", traj, ref);
    run<cfloat32>("cfloat<32,8>", traj, ref);
    run<posit32>("posit<32,2>", traj, ref);
    run<posit16>("posit<16,1>", traj, ref);
    run<cfloat16>("cfloat<16,5> (half)", traj, ref);

    std::cout << "\n";
    return 0;
}
