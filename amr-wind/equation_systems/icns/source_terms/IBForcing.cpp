#include "amr-wind/equation_systems/icns/source_terms/IBForcing.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/immersed_boundary/IB.H"
#include "amr-wind/equation_systems/PDEHelpers.H"
#include "amr-wind/core/vs/vector_space.H"
#include <AMReX_REAL.H>

#include "AMReX_Gpu.H"

#include <AMReX_FArrayBox.H> // this gets me abs(), but I need to follow it
// down.
//#include <AMReX_Math.H>

#include "amr-wind/immersed_boundary/complex_terrain/ComplexTerrain.H"
#include "amr-wind/immersed_boundary/IBOps.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/fvm/gradient.H"
#include "AMReX_Vector.H"
#include "amr-wind/wind_energy/actuator/actuator_utils.H"
#include "amr-wind/utilities/tensor_ops.H"

#include "AMReX_MultiFabUtil.H"

namespace amr_wind {
namespace pde {
namespace icns {
const std::string var_name = "velocity";

IBForcing::IBForcing(const CFDSim& sim)
    : m_ib_src(sim.repo().get_field("ib_src_term"))
    , m_ib_normal(sim.repo().get_field("ib_normal"))
    , m_diffterm(sim.repo().get_field(pde_impl::diff_term_name(var_name)))
{
    if (!sim.physics_manager().contains("IB")) {
        amrex::Abort("IBForcing requires IB physics to be active");
    }
}

IBForcing::~IBForcing() = default;

void IBForcing::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState /*fstate*/,
    const amrex::Array4<amrex::Real>& src_term) const
{
    const auto varr = m_ib_src(lev).const_array(mfi);
    const auto diffterm = m_diffterm(lev).const_array(mfi);
    const auto normal = m_ib_normal(lev).const_array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        // subtract the diffusion term from the forcing to zero viscous
        // and turbulent stresses in locations where where the wall
        // model is active (ib_normal is only non-zero where the wall
        // model is active)
        const vs::Vector phi_norm{
            normal(i, j, k, 0), normal(i, j, k, 1), normal(i, j, k, 2)};

        const amrex::Real mag_phi_norm = vs::mag(phi_norm);

        for (int ii = 0; ii < 3; ii++) {
            const amrex::Real viscous_delta =
                diffterm(i, j, k, ii) * mag_phi_norm - varr(i, j, k, ii);
            // const amrex::Real viscous_nudge =
            //     viscous_delta < 0 ? viscous_delta : 0.0;

            /* if (k == 3 && i == 0 && j == 0 && ii == 0) {
                amrex::Print() << "src: " << varr(i, j, k, ii)
                               << " diffterm: " << diffterm(i, j, k, ii)
                               << "  viscous_delta: " << viscous_delta
                               << "  viscous_nudge: " << viscous_nudge
                               << "  actual addition: "
                               << varr(i, j, k, ii) + viscous_nudge -
                                      diffterm(i, j, k, ii) * mag_phi_norm
                               << std::endl;
            }
 */
            // nmatula The if statement here will do the same thing as the old
            // viscous nudge.  Without the if, we'll just always be using the
            // wall model contribution in the band

            // if (amrex::Math::abs(diffterm(i, j, k, ii) * mag_phi_norm) <
            //     amrex::Math::abs(varr(i, j, k, ii))) {
            src_term(i, j, k, ii) += -viscous_delta;
            //}

            // src_term(i, j, k, ii) += viscous_nudge - viscous_delta;
        }
    });
}

} // namespace icns
} // namespace pde
} // namespace amr_wind