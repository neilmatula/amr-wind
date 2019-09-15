#include <AMReX_Array.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <incflo.H>
#include <boundary_conditions_F.H>
#include <setup_F.H>

//
// Fill the BCs for velocity only
//
void incflo::FillVelocityBC(Real time, int extrap_dir_bcs)
{
    BL_PROFILE("incflo::FillVelocityBC()");

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Box domain(geom[lev].Domain());

        // Hack so that ghost cells are not undefined
        vel[lev]->setDomainBndry(boundary_val, geom[lev]);

        vel[lev]->FillBoundary(geom[lev].periodicity());
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi(*vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            set_velocity_bcs(&time, 
                             BL_TO_FORTRAN_ANYD((*vel[lev])[mfi]),
                             bc_ilo[lev]->dataPtr(), bc_ihi[lev]->dataPtr(),
                             bc_jlo[lev]->dataPtr(), bc_jhi[lev]->dataPtr(),
                             bc_klo[lev]->dataPtr(), bc_khi[lev]->dataPtr(),
                             domain.loVect(), domain.hiVect(),
                             &nghost, &extrap_dir_bcs, &probtype);
        }
        EB_set_covered(*vel[lev], covered_val);
        
        // Do this after as well as before to pick up terms that got updated in the call above
        vel[lev]->FillBoundary(geom[lev].periodicity());
    }
}

void incflo::FillScalarBC()
{
    BL_PROFILE("incflo:FillScalarBC()");

    for(int lev = 0; lev <= finest_level; lev++)
    {
        Box domain(geom[lev].Domain());
        
        // Hack so that ghost cells are not undefined
        density[lev]->setDomainBndry(boundary_val, geom[lev]);
        tracer[lev]->setDomainBndry(boundary_val, geom[lev]);
        eta[lev]->setDomainBndry(boundary_val, geom[lev]);

        // Impose periodic BCs at domain boundaries and fine-fine copies in the interior
        density[lev]->FillBoundary(geom[lev].periodicity());
        tracer[lev]->FillBoundary(geom[lev].periodicity());
        eta[lev]->FillBoundary(geom[lev].periodicity());

        // Fill all cell-centered arrays with first-order extrapolation at domain boundaries
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi(*density[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Density
            fill_bc0(BL_TO_FORTRAN_ANYD((*density[lev])[mfi]),
                     bc_ilo[lev]->dataPtr(), bc_ihi[lev]->dataPtr(),
                     bc_jlo[lev]->dataPtr(), bc_jhi[lev]->dataPtr(),
                     bc_klo[lev]->dataPtr(), bc_khi[lev]->dataPtr(),
                     domain.loVect(), domain.hiVect(),
                     &nghost);

            // Tracer
            fill_bc0(BL_TO_FORTRAN_ANYD((*tracer[lev])[mfi]),
                     bc_ilo[lev]->dataPtr(), bc_ihi[lev]->dataPtr(),
                     bc_jlo[lev]->dataPtr(), bc_jhi[lev]->dataPtr(),
                     bc_klo[lev]->dataPtr(), bc_khi[lev]->dataPtr(),
                     domain.loVect(), domain.hiVect(),
                     &nghost);

            // Viscosity
            fill_bc0(BL_TO_FORTRAN_ANYD((*eta[lev])[mfi]),
                     bc_ilo[lev]->dataPtr(), bc_ihi[lev]->dataPtr(),
                     bc_jlo[lev]->dataPtr(), bc_jhi[lev]->dataPtr(),
                     bc_klo[lev]->dataPtr(), bc_khi[lev]->dataPtr(),
                     domain.loVect(), domain.hiVect(),
                     &nghost);
        }
    }
}

void incflo::GetInputBCs()
{
    // Extracts all walls from the inputs file
    int cyclic;

    cyclic = geom[0].isPeriodic(0) ? 1 : 0;
    SetInputBCs("xlo", 1, cyclic, geom[0].ProbLo(0));
    SetInputBCs("xhi", 2, cyclic, geom[0].ProbHi(0));

    cyclic = geom[0].isPeriodic(1) ? 1 : 0;
    SetInputBCs("ylo", 3, cyclic, geom[0].ProbLo(1));
    SetInputBCs("yhi", 4, cyclic, geom[0].ProbHi(1));

    cyclic = geom[0].isPeriodic(2) ? 1 : 0;
    SetInputBCs("zlo", 5, cyclic, geom[0].ProbLo(2));
    SetInputBCs("zhi", 6, cyclic, geom[0].ProbHi(2));
}

void incflo::SetInputBCs(const std::string bcID, const int index,
                           const int cyclic, const Real domloc) 
{
    const int und_  =   0;
    const int pinf_ =  10;
    const int pout_ =  11;
    const int minf_ =  20;
    const int nsw_  = 100;

    // Default a BC to undefined.
    int itype = und_;

    int direction = 0;
    Real mi_pressure = -1.0;
    Real mi_density =  1.0;
    Real mi_tracer =  1.0;
    Vector<Real> mi_velocity(3, 0.0);
    Real location = domloc;

    std::string bc_type = "null";

    ParmParse pp(bcID);

    pp.query("type", bc_type);

    if(bc_type == "pressure_inflow"  || bc_type == "pi" ||
              bc_type == "PRESSURE_INFLOW"  || bc_type == "PI" ) {

      amrex::Print() << bcID <<" set to pressure inflow. "  << std::endl;
      itype = pinf_;

      pp.get("pressure", mi_pressure);

    } else if(bc_type == "pressure_outflow" || bc_type == "po" ||
              bc_type == "PRESSURE_OUTFLOW" || bc_type == "PO" ) {

      amrex::Print() << bcID <<" set to pressure outflow. "  << std::endl;
      itype = pout_;

      pp.get("pressure", mi_pressure);


    } else if (bc_type == "mass_inflow"     || bc_type == "mi" ||
               bc_type == "MASS_INFLOW"     || bc_type == "MI" ) {

      // Flag that this is a mass inflow.
      amrex::Print() << bcID <<" set to mass inflow. "  << std::endl;
      itype = minf_;

      pp.query("pressure", mi_pressure);
      pp.getarr("velocity", mi_velocity, 0, 3);

      pp.query("density", mi_density);
      pp.query("tracer", mi_tracer);

    } else if (bc_type == "no_slip_wall"    || bc_type == "nsw" ||
               bc_type == "NO_SLIP_WALL"    || bc_type == "NSW" ) {

      // Flag that this is a no-slip wall.
      amrex::Print() << bcID <<" set to no-slip wall. "  << std::endl;
      itype = nsw_;

      pp.queryarr("velocity", mi_velocity, 0, 3);
      pp.query("direction", direction);
      pp.query("location", location);

    }

    if ( cyclic == 1 && itype != und_){
      amrex::Abort("Cannot mix periodic BCs and Wall/Flow BCs.\n");
    }

    const Real* plo = geom[0].ProbLo();
    const Real* phi = geom[0].ProbHi();

    set_bc_mod(&index, &itype, plo, phi,
               &location, &mi_pressure, &mi_velocity[0], &mi_density, &mi_tracer);

}
