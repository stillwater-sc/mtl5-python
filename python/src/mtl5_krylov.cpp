// mtl5-python -- Krylov solver / preconditioner dispatch.
//
// MTL5 has 10 Krylov solvers and 8 preconditioners, and every solver takes the
// preconditioner as a template parameter. Binding the cross product directly
// would be 10 x 8 x 2 dtypes = 160 instantiations of a full iterative solver.
//
// Instead the preconditioner is type-erased once. `PCBase<T>` is an abstract
// base with virtual apply/apply_adjoint; `PCRef<T>` is the thin value adapter
// that satisfies MTL5's `M.solve(x, b)` / `M.adjoint_solve(x, b)` contract by
// forwarding through that base. Every solver is then instantiated exactly once
// per element type against `PCRef<T>` -- 10 x 2 = 20 -- and any preconditioner
// can be handed to any solver at runtime. The virtual call costs nothing
// measurable next to a sparse triangular solve.
//
// It also hides an upstream inconsistency that would otherwise leak into the
// Python surface: `identity`, `diagonal`, `ssor` and `block_diagonal` are
// templated on the *matrix* type while `ilu_0`, `ic_0`, `ilut` and `ildl` are
// templated on the *value* type.
//
// **bicg and qmr need a symmetric preconditioner.** They are the only two that
// call `M.adjoint_solve`, and every MTL5 preconditioner implements it as
// `solve` -- commented "same as solve (approximate)". That is exact when the
// preconditioner is symmetric and wrong when it is not, so with a non-symmetric
// ILU(0) or SSOR they break down rather than converge. Measured on a
// non-symmetric 144x144 system: bicg + ilu_0 reaches rel_err 2.9e+00, and on a
// symmetric one the same pair converges in 20 iterations to 1.7e-12. So each
// preconditioner records whether it is symmetric, and bicg/qmr refuse the ones
// that are not rather than returning a plausible wrong answer.
//
// (Two separate upstream bugs in this area were found and fixed while building
// this: stillwater-sc/mtl5#392 -- bicg computed rho from the wrong vectors and
// worked only with the identity preconditioner -- and the qmr preconditioner
// application it turned up alongside.)

#include "mtl5_types.hpp"
#include "mtl5_sparse_refine.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicg.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/bicgstab_ell.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/cgs.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/krylov/idr_s.hpp>
#include <mtl/itl/krylov/minres.hpp>
#include <mtl/itl/krylov/qmr.hpp>
#include <mtl/itl/krylov/tfqmr.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/operation/matrix_properties.hpp>

#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <tuple>

using namespace nb::literals;

namespace {

namespace pc = mtl::itl::pc;

template <typename T> using SMat = mtl::mat::compressed2D<T>;
template <typename T> using Vec  = mtl::vec::dense_vector<T>;

// ---------------------------------------------------------------------------
// Type-erased preconditioner
// ---------------------------------------------------------------------------
template <typename T>
struct PCBase {
    virtual ~PCBase() = default;
    virtual void apply(Vec<T>& x, const Vec<T>& b) const = 0;
    virtual void apply_adjoint(Vec<T>& x, const Vec<T>& b) const = 0;
    /// Whether M == M^T. bicg and qmr are only correct when this holds, because
    /// MTL5's adjoint_solve is an alias for solve.
    virtual bool is_symmetric() const = 0;
    virtual std::size_t size() const = 0;
    virtual const char* kind() const = 0;
};

/// The value type MTL5's solvers see. Forwards to the erased base, so each
/// solver is instantiated once per element type rather than once per pairing.
template <typename T>
struct PCRef {
    const PCBase<T>* p;
    template <typename X, typename B> void solve(X& x, const B& b) const {
        p->apply(x, b);
    }
    template <typename X, typename B> void adjoint_solve(X& x, const B& b) const {
        p->apply_adjoint(x, b);
    }
};

template <typename PCType, typename T>
struct PCImpl : PCBase<T> {
    PCType pc_;
    std::size_t n;          // named for sparse_refine_through, which reads M.n
    bool sym_;
    const char* kind_;

    template <typename... Args>
    PCImpl(const SMat<T>& A, bool sym, const char* kind, Args&&... args)
        : pc_(A, std::forward<Args>(args)...), n(A.num_rows()), sym_(sym), kind_(kind) {}

    void apply(Vec<T>& x, const Vec<T>& b) const override { pc_.solve(x, b); }
    void apply_adjoint(Vec<T>& x, const Vec<T>& b) const override {
        pc_.adjoint_solve(x, b);
    }
    bool is_symmetric() const override { return sym_; }
    std::size_t size() const override { return n; }
    const char* kind() const override { return kind_; }

    // Kept so mtl5.mixed.iterative_refine, which drives a factorization through
    // .factor().solve(x, b), can take a preconditioner unchanged.
    const PCType& factor() const { return pc_; }
};

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------
enum class Solver {
    cg, bicg, bicgstab, bicgstab_ell, cgs, gmres, idr_s, minres, qmr, tfqmr
};

Solver parse_solver(const std::string& s) {
    if (s == "cg")           return Solver::cg;
    if (s == "bicg")         return Solver::bicg;
    if (s == "bicgstab")     return Solver::bicgstab;
    if (s == "bicgstab_ell") return Solver::bicgstab_ell;
    if (s == "cgs")          return Solver::cgs;
    if (s == "gmres")        return Solver::gmres;
    if (s == "idr_s")        return Solver::idr_s;
    if (s == "minres")       return Solver::minres;
    if (s == "qmr")          return Solver::qmr;
    if (s == "tfqmr")        return Solver::tfqmr;
    throw std::invalid_argument(
        "unknown solver '" + s + "'; valid: cg, bicg, bicgstab, bicgstab_ell, "
        "cgs, gmres, idr_s, minres, qmr, tfqmr");
}

/// bicg and qmr are the only solvers that use the adjoint of the
/// preconditioner, and MTL5 supplies an approximation rather than the real
/// thing. Refuse rather than return the plausible-looking wrong answer that
/// combination produces.
void require_symmetric_pc(Solver s, const PCBase<double>* pcd,
                          const PCBase<float>* pcf, const std::string& name) {
    const bool sym = pcd ? pcd->is_symmetric() : pcf->is_symmetric();
    const char* kind = pcd ? pcd->kind() : pcf->kind();
    if (sym) return;
    if (s != Solver::bicg && s != Solver::qmr) return;
    throw std::invalid_argument(
        name + " needs a symmetric preconditioner, and '" + kind + "' is not "
        "symmetric for this matrix. It is the only kind of solver here that "
        "applies M^T, and MTL5's preconditioners implement adjoint_solve as "
        "solve -- exact when M is symmetric, wrong otherwise, and it breaks "
        "down rather than converging. Use bicgstab, gmres, cgs, idr_s, "
        "bicgstab_ell or tfqmr with this preconditioner, or pair bicg/qmr with "
        "identity, diagonal, ic0 or ildl.");
}

template <typename T>
std::tuple<VectorView<T>, int, int, double>
run_solver(Solver which, const SMat<T>& A, const PCBase<T>& M,
           const Vec<T>& b, T rtol, int maxiter, int restart, int ell, int s)
{
    Vec<T> x(A.num_rows(), T{0});
    mtl::itl::basic_iteration<T> iter(b, maxiter, rtol);
    const PCRef<T> Mref{&M};
    int info = 0;
    {
        nogil guard;
        switch (which) {
            case Solver::cg:       info = mtl::itl::cg(A, x, b, Mref, iter); break;
            case Solver::bicg:     info = mtl::itl::bicg(A, x, b, Mref, iter); break;
            case Solver::bicgstab: info = mtl::itl::bicgstab(A, x, b, Mref, iter); break;
            case Solver::cgs:      info = mtl::itl::cgs(A, x, b, Mref, iter); break;
            case Solver::minres:   info = mtl::itl::minres(A, x, b, Mref, iter); break;
            case Solver::qmr:      info = mtl::itl::qmr(A, x, b, Mref, iter); break;
            case Solver::tfqmr:    info = mtl::itl::tfqmr(A, x, b, Mref, iter); break;
            case Solver::gmres:    info = mtl::itl::gmres(A, x, b, Mref, iter, restart); break;
            case Solver::idr_s:    info = mtl::itl::idr_s(A, x, b, Mref, iter, s); break;
            case Solver::bicgstab_ell:
                info = mtl::itl::bicgstab_ell(A, x, b, Mref, iter, ell); break;
        }
    }
    return {VectorView<T>(std::move(x)), info, static_cast<int>(iter.iterations()),
            static_cast<double>(iter.resid())};
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
template <typename T>
void register_preconditioners(nb::module_& m) {
    const std::string suffix = std::string("_") + type_suffix<T>();

    nb::class_<PCBase<T>>(m, ("Preconditioner" + suffix).c_str(),
        "Base class for the preconditioners accepted as M= by the Krylov solvers")
        .def_prop_ro("n", [](const PCBase<T>& p) { return p.size(); })
        .def_prop_ro("kind", [](const PCBase<T>& p) { return p.kind(); })
        .def_prop_ro("is_symmetric", [](const PCBase<T>& p) { return p.is_symmetric(); },
                     "Whether M == M^T. bicg and qmr require it, because MTL5 "
                     "implements adjoint_solve as solve.")
        .def("solve", [](const PCBase<T>& p, const VectorView<T>& b) {
            if (b.vec.size() != p.size())
                throw std::invalid_argument("solve: dimension mismatch");
            Vec<T> x(p.size()), bb(p.size());
            for (std::size_t i = 0; i < p.size(); ++i) bb[i] = b.vec[i];
            {
                nogil guard;
                p.apply(x, bb);
            }
            return VectorView<T>(std::move(x));
        }, "b"_a, "Apply the preconditioner: solve M x = b")
        .def("solve", [](const PCBase<T>& p,
                         nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) {
            if (b.shape(0) != p.size())
                throw std::invalid_argument("solve: dimension mismatch");
            Vec<T> x(p.size()), bb(p.size());
            for (std::size_t i = 0; i < p.size(); ++i) bb[i] = b.data()[i];
            {
                nogil guard;
                p.apply(x, bb);
            }
            return VectorView<T>(std::move(x));
        }, "b"_a)
        // Report the concrete class (ILU0_f64, SSOR_f64, ...) rather than the
        // erased base, which is what a caller sees and constructs.
        .def("__repr__", [](nb::handle self) {
            const PCBase<T>& p = nb::cast<const PCBase<T>&>(self);
            std::string cls = nb::cast<std::string>(self.type().attr("__name__"));
            return "mtl5.sparse." + cls + "(n=" + std::to_string(p.size()) +
                   (p.is_symmetric() ? ", symmetric" : ", non-symmetric") + ")";
        });
}

void require_square(const SMat<double>&) {}

/// Whether this preconditioner will be symmetric. For the kinds whose structure
/// follows A, that is a property of A and is settled once, at construction.
template <typename T>
bool pc_symmetry(const SMat<T>& A, bool always) {
    if (always) return true;
    nogil guard;
    return mtl::is_symmetric(A);
}

template <typename PCType, typename T>
nb::class_<PCImpl<PCType, T>, PCBase<T>> pc_class(nb::module_& m,
                                                  const char* py_name,
                                                  const char* doc) {
    const std::string name = std::string(py_name) + "_" + type_suffix<T>();
    return nb::class_<PCImpl<PCType, T>, PCBase<T>>(m, name.c_str(), doc);
}

/// The single-argument kinds: constructed from A alone.
template <typename PCType, typename T>
void bind_simple_pc(nb::module_& m, const char* py_name, const char* kind,
                    bool always_symmetric, const char* doc) {
    using Impl = PCImpl<PCType, T>;
    pc_class<PCType, T>(m, py_name, doc)
        .def("__init__", [kind, always_symmetric](Impl* self, const SMat<T>& A) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("preconditioner: matrix must be square");
            const bool sym = pc_symmetry(A, always_symmetric);
            nogil guard;
            new (self) Impl(A, sym, kind);
        }, "A"_a);
}

template <typename T>
void register_solvers(nb::module_& m) {
    m.def("_krylov", [](const std::string& solver, const SMat<T>& A,
                        const PCBase<T>& M, const VectorView<T>& b,
                        double rtol, int maxiter, int restart, int ell, int s) {
        const Solver which = parse_solver(solver);
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument(solver + ": A must be square");
        if (A.num_rows() != b.vec.size())
            throw std::invalid_argument(solver + ": A.num_rows must equal len(b)");
        if (M.size() != A.num_rows())
            throw std::invalid_argument(
                solver + ": preconditioner size " + std::to_string(M.size()) +
                " does not match A (" + std::to_string(A.num_rows()) + ")");
        if constexpr (std::is_same_v<T, double>)
            require_symmetric_pc(which, &M, nullptr, solver);
        else
            require_symmetric_pc(which, nullptr, &M, solver);

        Vec<T> bb(b.vec.size());
        for (std::size_t i = 0; i < b.vec.size(); ++i) bb[i] = b.vec[i];
        return run_solver<T>(which, A, M, bb, static_cast<T>(rtol), maxiter,
                             restart, ell, s);
    }, "solver"_a, "A"_a, "M"_a, "b"_a, "rtol"_a, "maxiter"_a,
       "restart"_a = 30, "ell"_a = 2, "s"_a = 4,
       "Internal Krylov dispatch — use mtl5.sparse.cg() and friends");
}

template <typename T>
void register_for(nb::module_& m) {
    register_preconditioners<T>(m);

    // Structurally symmetric regardless of A.
    bind_simple_pc<pc::identity<SMat<T>>, T>(m, "Identity", "identity", true,
        "Identity preconditioner -- a no-op, for running a solver unpreconditioned");
    bind_simple_pc<pc::diagonal<SMat<T>>, T>(m, "Diagonal", "diagonal", true,
        "Jacobi / diagonal preconditioner");
    bind_simple_pc<pc::ic_0<T>, T>(m, "IC0", "ic0", true,
        "Incomplete Cholesky with no fill-in, for symmetric positive definite A");
    bind_simple_pc<pc::ildl<T>, T>(m, "ILDL", "ildl", true,
        "Incomplete LDL^T, for symmetric indefinite A");

    // Symmetric only when A is; settled at construction.
    bind_simple_pc<pc::ilu_0<T>, T>(m, "ILU0", "ilu0", false,
        "Incomplete LU with no fill-in");

    using SSORImpl = PCImpl<pc::ssor<SMat<T>>, T>;
    pc_class<pc::ssor<SMat<T>>, T>(m, "SSOR",
        "Symmetric successive over-relaxation")
        .def("__init__", [](SSORImpl* self, const SMat<T>& A, double omega) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SSOR: matrix must be square");
            if (!(omega > 0.0 && omega < 2.0))
                throw nb::value_error("SSOR: omega must lie in (0, 2)");
            const bool sym = pc_symmetry(A, false);
            nogil guard;
            new (self) SSORImpl(A, sym, "ssor", static_cast<T>(omega));
        }, "A"_a, "omega"_a = 1.0);

    using ILUTImpl = PCImpl<pc::ilut<T>, T>;
    pc_class<pc::ilut<T>, T>(m, "ILUT",
        "Incomplete LU with a drop threshold and a fill limit per row")
        .def("__init__", [](ILUTImpl* self, const SMat<T>& A, std::size_t fill,
                            double tau) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("ILUT: matrix must be square");
            const bool sym = pc_symmetry(A, false);
            nogil guard;
            new (self) ILUTImpl(A, sym, "ilut", fill, static_cast<T>(tau));
        }, "A"_a, "fill"_a = 10, "tau"_a = 1e-4);

    using BDImpl = PCImpl<pc::block_diagonal<SMat<T>>, T>;
    pc_class<pc::block_diagonal<SMat<T>>, T>(m, "BlockDiagonal",
        "Block Jacobi with a fixed block size")
        .def("__init__", [](BDImpl* self, const SMat<T>& A, std::size_t block_size) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("BlockDiagonal: matrix must be square");
            if (block_size == 0)
                throw nb::value_error("BlockDiagonal: block_size must be >= 1");
            const bool sym = pc_symmetry(A, false);
            nogil guard;
            new (self) BDImpl(A, sym, "block_diagonal", block_size);
        }, "A"_a, "block_size"_a);

    register_solvers<T>(m);
}

}  // namespace

// ===========================================================================
void register_krylov(nb::module_& m) {
    register_for<float>(m);
    register_for<double>(m);

    m.def("solvers", []() {
        return std::vector<std::string>{
            "cg", "bicg", "bicgstab", "bicgstab_ell", "cgs",
            "gmres", "idr_s", "minres", "qmr", "tfqmr"};
    }, "The Krylov solvers available to mtl5.sparse.iterative_solve()");

    // mtl5.mixed.iterative_refine refines through anything exposing .n and
    // .factor().solve(x, b). ILU(0) and IC(0) moved into this TU, so their
    // overloads are re-registered here rather than left dangling.
    nb::module_ mx = nb::borrow<nb::module_>(m.attr("mixed"));
    mx.def("iterative_refine",
           &sparse_refine_through<PCImpl<pc::ilu_0<double>, double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine",
           &sparse_refine_through<PCImpl<pc::ic_0<double>, double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);

    m.def("preconditioners", []() {
        return std::vector<std::string>{
            "identity", "diagonal", "ic0", "ildl",
            "ilu0", "ssor", "ilut", "block_diagonal"};
    }, "The preconditioners accepted as M=");
}
