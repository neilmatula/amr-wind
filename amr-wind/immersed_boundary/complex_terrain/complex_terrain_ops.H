#ifndef COMPLEX_TERRAIN_OPS_H
#define COMPLEX_TERRAIN_OPS_H

#include "amr-wind/immersed_boundary/complex_terrain/ComplexTerrain.H"
#include "amr-wind/immersed_boundary/IBOps.H"
#include "amr-wind/core/MultiParser.H"
#include "amr-wind/fvm/gradient.H"
#include "AMReX_Vector.H"
#include "amr-wind/wind_energy/actuator/actuator_utils.H"
#include "amr-wind/utilities/tensor_ops.H"

#include "AMReX_MultiFabUtil.H"

namespace amr_wind {
namespace ib {
namespace complex_terrain {

/** Parse common data for modeling non-blade bluff bodies via an immersed
 * bounary
 */
void read_inputs(
    ComplexTerrainBaseData&, IBInfo&, const ::amr_wind::utils::MultiParser&);

/** Initialize core data structures when modeling the IB geometry
 */
void init_data_structures(CFDSim&);

/** Set the velocity inside the IB Complex Terrain
 */
void apply_dirichlet_vel(CFDSim&, const amrex::Vector<amrex::Real>& vel_bc);

void compute_wall_stresses(
    CFDSim& sim, const amrex::Vector<amrex::Real>& tau_wall);

void prepare_netcdf_file(
    const std::string&, const ComplexTerrainBaseData&, const IBInfo&);

void write_netcdf(
    const std::string&,
    const ComplexTerrainBaseData&,
    const IBInfo&,
    const amrex::Real);

} // namespace complex_terrain

namespace ops {

template <typename GeomTrait>
struct UseDefaultOp
{
    static constexpr bool update_pos = true;
    static constexpr bool update_vel = true;
    static constexpr bool compute_force = true;
    static constexpr bool process_outputs = true;
};

template <typename GeomTrait>
struct UpdatePosOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
    void operator()(typename GeomTrait::DataType&) {}
};

template <typename GeomTrait>
struct UpdateVelOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
    void operator()(typename GeomTrait::DataType& data)
    {
        BL_PROFILE("amr-wind::ib::IB::update_velocities");

        const auto& wdata = data.meta();
        auto& sim = data.sim();

        complex_terrain::apply_dirichlet_vel(sim, wdata.vel_bc);
    }
};

template <typename GeomTrait>
struct ComputeForceOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
    void operator()(typename GeomTrait::DataType&) {}
};

template <typename IbType>
struct ComputeSourceOp<
    IbType,
    std::enable_if_t<std::is_base_of<ComplexTerrainType, IbType>::value>>
{
    void operator()(
        typename IbType::DataType& data,
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Geometry& geom)
    {
        Field& source = data.sim().repo().get_field("ib_src_term");
        const Field& level = data.sim().repo().get_field("ib_levelset");
        const Field& normal = data.sim().repo().get_field("ib_normal");
        const Field& velocity = data.sim().repo().get_field("velocity");
        const Field& density = data.sim().repo().get_field("density");

        const auto& bx = mfi.tilebox();
        const auto& varr = velocity(lev).array(mfi);
        const auto& darr = density(lev).array(mfi);
        const auto& phi_arr = level(lev).array(mfi);
        const auto& norm_arr = normal(lev).array(mfi);
        auto sarr = source(lev).array(mfi);
        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();

        const amrex::Real cd = data.meta().drag_coefficient;
        const amrex::Real eps = data.meta().eps_spread;

        const amrex::Real h = dx[2];

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Vector<amrex::Real> vel = {
                    {varr(i, j, k, 0), varr(i, j, k, 1), varr(i, j, k, 2)}};
                const amrex::Vector<amrex::Real> norm = {
                    {norm_arr(i, j, k, 0), norm_arr(i, j, k, 1),
                     norm_arr(i, j, k, 2)}};

                /* const vs::Vector cc{
                    problo[0] + (i + 0.5) * dx[0],
                    problo[1] + (j + 0.5) * dx[1],
                    problo[2] + (k + 0.5) * dx[2],
                }; */

                // Outside the band if the normal vector is assigned to be
                // zero normal vector will only be ~1 or ~0 so 0.5 is a safe
                // value that to confirm we are in the band where the wall
                // model is active
                const amrex::Real nmag = utils::vec_mag(norm.data());
                if (nmag > 0.5) {

                    // nmatula todo
                    // account for the case where the point below would be below
                    // the bottom boundary

                    const amrex::Real z_cell_centroid =
                        problo[2] + (k + 0.5) * dx[2];
                    const amrex::Real phi_cell_centroid = phi_arr(i, j, k);
                    // nmatula I did it like this because, when we do 3D curvy
                    // stuff, we're going to march from the cell centroid, in
                    // the direction of the normal, by a step equal to
                    // h - phi_cell_centroid
                    const amrex::Real z_outside_band =
                        z_cell_centroid + h - phi_cell_centroid;
                    // const amrex::Real z_outside_band =
                    //     z_cell_centroid + h / 2 + 0.00001;
                    //  nmatula todo use different floor function to avoid float
                    //  to int conversion
                    const int bounding_cell =
                        floor((z_outside_band - problo[2]) / h);
                    const amrex::Real z_bounding_cell =
                        problo[2] + (bounding_cell + 0.5) * dx[2];
                    const amrex::Real u_bounding_cell =
                        varr(i, j, bounding_cell, 0);

                    // Linear interpolation in z for now
                    // Will probably be trilinear interpolation in the general
                    // case (hill)
                    amrex::Real u_interp;
                    if (z_bounding_cell >= z_outside_band) {
                        const amrex::Real z_below =
                            problo[2] + (bounding_cell - 0.5) * dx[2];
                        const amrex::Real u_below =
                            varr(i, j, bounding_cell - 1, 0);
                        u_interp = u_below + (z_outside_band - z_below) *
                                                 (u_bounding_cell - u_below) /
                                                 (z_bounding_cell - z_below);
                    } else {
                        const amrex::Real z_above =
                            problo[2] + (bounding_cell + 1.5) * dx[2];
                        const amrex::Real u_above =
                            varr(i, j, bounding_cell + 1, 0);
                        u_interp = u_bounding_cell +
                                   (z_outside_band - z_bounding_cell) *
                                       (u_above - u_bounding_cell) /
                                       (z_above - z_bounding_cell);
                    }

                    // debug output
                    if (1 == 0) {
                        std::cerr << "h:                   " << h << std::endl;
                        std::cerr
                            << "phi_cell_centroid:   " << phi_cell_centroid
                            << std::endl;
                        std::cerr << "z_cell_centroid:     " << z_cell_centroid
                                  << std::endl;
                        std::cerr << "z_outside_band:      " << z_outside_band
                                  << std::endl;
                        std::cerr << "IB height:           "
                                  << z_cell_centroid - phi_cell_centroid
                                  << std::endl;
                        std::cerr << "bounding cell index: " << bounding_cell
                                  << std::endl;
                        std::cerr << "band cell index      " << k << std::endl;

                        // exit(1);
                    }

                    const amrex::Real vdotn =
                        utils::dot_prod(norm.data(), vel.data());
                    const auto gauss_fac = actuator::utils::gaussian1d(
                        amrex::Math::abs(phi_arr(i, j, k)), eps);
                    // v_tan = v - (v dot n) n
                    amrex::Vector<amrex::Real> vtan = {
                        {vel[0] - vdotn * norm[0], vel[1] - vdotn * norm[1],
                         vel[2] - vdotn * norm[2]}};
                    // Get tangent velocity magnitude
                    const amrex::Real vtmag = utils::vec_mag(vtan.data());

                    // Old thing
                    // const amrex::Real divt = -cd / vtmag;

                    // todo
                    // Hard coded random z0 for now
                    // Can either pull from ABL z0 in input file, or have an
                    // IB-specific parameter in input file
                    const amrex::Real z0 = 0.000001;

                    // todo pull kappa from correct place
                    const amrex::Real kappa = 0.4;

                    // Still got the factor of vtmag because vtan is
                    // unnormalized
                    // (divt is a bad name)
                    const amrex::Real divt =
                        -2.0 / phi_arr(i, j, k) /
                        pow(log(phi_arr(i, j, k) / z0), 3) * kappa * kappa *
                        u_interp * u_interp / vtmag;
                    sarr(i, j, k, 0) += divt * vtan[0];
                    sarr(i, j, k, 1) += divt * vtan[1];
                    sarr(i, j, k, 2) += divt * vtan[2];
                }
            });
    }
};

template <typename GeomTrait>
struct ProcessOutputsOp<
    GeomTrait,
    typename std::enable_if<
        std::is_base_of<ComplexTerrainType, GeomTrait>::value>::type>
{
private:
    typename GeomTrait::DataType& m_data;

    std::string m_out_dir{""};
    std::string m_nc_filename{""};
    int m_out_freq{10};

public:
    explicit ProcessOutputsOp(typename GeomTrait::DataType& data) : m_data(data)
    {}

    void read_io_options(const ::amr_wind::utils::MultiParser& pp)
    {
        pp.query("output_frequency", m_out_freq);
    }

    void prepare_outputs(const std::string& out_dir)
    {
        m_nc_filename = out_dir + "/" + m_data.info().label + ".nc";
        complex_terrain::prepare_netcdf_file(
            m_nc_filename, m_data.meta(), m_data.info());
    }

    void write_outputs()
    {
        const auto& time = m_data.sim().time();
        const int tidx = time.time_index();
        if (tidx % m_out_freq != 0) return;

        complex_terrain::write_netcdf(
            m_nc_filename, m_data.meta(), m_data.info(), time.new_time());
    }
};

} // namespace ops

} // namespace ib
} // namespace amr_wind

#endif /* COMPLEX_TERRAIN_OPS_H */
