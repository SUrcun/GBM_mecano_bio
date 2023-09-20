// Copyright (C) 2006-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-02-09
// Last changed: 2012-07-05
//


#include <dolfin.h>
#include "Glioma.h"
#include "Poro.h"
#include "Disp.h"
#include "Reg_w.h"
#include "Necro.h"
#include "Reg_nec.h"
#include "Rho.h"
#include "Reg_rho.h"
#include "Young.h"
#include "RT_TMZ.h"
#include "RT_TMZ_TC.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/io/File.h>
#include <dolfin/io/HDF5File.h>
#include <dolfin/io/HDF5Interface.h>

using namespace dolfin;


void update_solution(Function& a, const Function& w, const Function& w0)
{
  *a.vector() += *w.vector();
  *a.vector() += *w0.vector();
   a.vector()->apply("add");
   as_type<PETScVector>(a.vector())->update_ghost_values();

};

void update_young(Function& a_young, const Function& d_young, const Function& e_young)
{
  *a_young.vector() += *d_young.vector();
  *a_young.vector() += *e_young.vector();
   a_young.vector()->apply("add");
   as_type<PETScVector>(a_young.vector())->update_ghost_values();

};


class on_bound : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return on_boundary; }
};


int main()
{
  parameters["std_out_all_processes"] = false;
  
  auto mesh = std::make_shared<Mesh>();
  
  File file_mesh(mesh->mpi_comm(), "./MESH_reduced/mesh.xml.gz");
  file_mesh >> *mesh; 
   
  double t = 0.0; 
  
  // Create function spaces
  auto V = std::make_shared<Glioma::FunctionSpace>(mesh);
  auto V1 = std::make_shared<Poro::FunctionSpace>(mesh);
  auto V2 = std::make_shared<Disp::FunctionSpace>(mesh);
  auto W1 = std::make_shared<Reg_w::FunctionSpace>(mesh);
  auto W2 = std::make_shared<Necro::FunctionSpace>(mesh);
  auto W3 = std::make_shared<Reg_nec::FunctionSpace>(mesh);
  auto W4 = std::make_shared<Rho::FunctionSpace>(mesh);
  auto W5 = std::make_shared<Reg_rho::FunctionSpace>(mesh);
  auto W6 = std::make_shared<Young::FunctionSpace>(mesh);
  auto W7 = std::make_shared<RT_TMZ::FunctionSpace>(mesh);
  auto W8 = std::make_shared<RT_TMZ_TC::FunctionSpace>(mesh);

  Timer timer("I/O time materials");
  //Material functions and internal variables
  auto necro_RT0 = std::make_shared<Function>(W7);
  auto e_young = std::make_shared<Function>(W6);
  auto diffusion = std::make_shared<Function>(V1);
  auto e_permea = std::make_shared<Function>(V1);
  auto poro0 = std::make_shared<Function>(V1);
  auto poro_b0 = std::make_shared<Function>(V1);
  auto t_wrho = std::make_shared<Function>(V1);
  auto wnecro0 = std::make_shared<Function>(W2);
  
  //Unknowns of the porous system
  auto w0 = std::make_shared<Function>(V);  
  auto u_n = std::make_shared<Function>(V2);

  File file_disp(mesh->mpi_comm(), "./DATA_reduced/disp.xml.gz"); 
  File file_e_young(mesh->mpi_comm(), "./DATA_reduced/young.xml.gz");
  File file_diffusion(mesh->mpi_comm(), "./DATA_reduced/diffusion.xml.gz");
  File file_e_permea(mesh->mpi_comm(), "./DATA_reduced/permea.xml.gz");  
  File file_poro(mesh->mpi_comm(), "./DATA_reduced/poro.xml.gz"); 
  File file_poro_b(mesh->mpi_comm(), "./DATA_reduced/poro_b.xml.gz"); 
  File file_wrho(mesh->mpi_comm(), "./DATA_reduced/wrho.xml.gz"); 
  File file_wnecro(mesh->mpi_comm(), "./DATA_reduced/wnecro.xml.gz");
  File file_w0(mesh->mpi_comm(), "./DATA_reduced/Xn.xml.gz"); 

  file_disp >> *u_n;
  file_e_young >> *e_young; 
  file_e_permea>> *e_permea; 
  file_diffusion>> *diffusion; 
  file_poro >> *poro0; 
  file_poro_b >> *poro_b0; 
  file_wrho >> *t_wrho;
  file_wnecro >> *wnecro0; 
  file_wnecro >> *necro_RT0;
  file_w0 >> *w0; 
  const double t_test = timer.stop();
  std::cout << "timer " << t_test << std::endl;

  //Unknowns
  auto w = std::make_shared<Function>(V);
  auto w_RT = std::make_shared<Function>(W8);
  auto poro = std::make_shared<Function>(V1);
  auto u = std::make_shared<Function>(V2);
  auto w_reg = std::make_shared<Function>(W1);
  auto wnecro = std::make_shared<Function>(W2);
  auto necro_RT = std::make_shared<Function>(W7);
  auto wnecro_reg = std::make_shared<Function>(W3);
  auto wrho = std::make_shared<Function>(W4);
  auto wrho_reg = std::make_shared<Function>(W5);
  auto d_young = std::make_shared<Function>(W6);

  ////Model parameters fixed
  int num_steps = 1009;
  auto dt = std::make_shared<Constant>(10.0);
  auto nu = std::make_shared<Constant>(0.47);
  auto mu_l = std::make_shared<Constant>(0.05);
  auto mu_h = std::make_shared<Constant>(35);  
  auto mu_t0 = std::make_shared<Constant>(35);  
  auto aECM = std::make_shared<Constant>(550);
  auto rho_l = std::make_shared<Constant>(1000.0);
  auto rho_t = std::make_shared<Constant>(1000.0);
  auto rho_h = std::make_shared<Constant>(1000.0);
  auto rho_s = std::make_shared<Constant>(1000.0);
  auto D_0 = std::make_shared<Constant>(2.5e-9);
  auto delta = std::make_shared<Constant>(2.0);
  auto w_env = std::make_shared<Constant>(1.9e-6);
  auto sig_th = std::make_shared<Constant>(12.0);
  auto w_art = std::make_shared<Constant>(1.9e-6);
  auto wcrit_angio = std::make_shared<Constant>(9.3e-7);  
  auto rate_wb0 = std::make_shared<Constant>(0.12);
  auto borne_pth = std::make_shared<Constant>(0.01);
  auto borne_wnl = std::make_shared<Constant>(0.0); 
  auto TMZ_t = std::make_shared<Constant>(0.06);
  auto rate_kill = std::make_shared<Constant>(0.99);
  auto mMGMT = std::make_shared<Constant>(0.5);
  auto poro_b_max = std::make_shared<Constant>(0.65);
  
  ////Model parameters for sensitivity
  //Mechanical
  auto p1 = std::make_shared<Constant>(910.0);
  auto p_crit = std::make_shared<Constant>(1500.0);
  auto k_max = std::make_shared<Constant>(1e-10);
  auto sig_hl = std::make_shared<Constant>(54.0);
  auto young_max = std::make_shared<Constant>(4000.0);
  //Biological (Oxygen)
  auto gam_tg0 = std::make_shared<Constant>(2.7e-2);
  auto gam_nl_t = std::make_shared<Constant>(4.1);
  auto gam_b_nl = std::make_shared<Constant>(0.0162);
  auto w_crit0 = std::make_shared<Constant>(7e-7);
  auto gam_tn = std::make_shared<Constant>(0.01); 
  auto w_crit_h = std::make_shared<Constant>(1e-6);
  auto gam_nl_h = std::make_shared<Constant>(0.25);  
  auto gam_csf_nl = std::make_shared<Constant>(0.04);
  //Coupled (Phenotype)
  auto IDH_threshold = std::make_shared<Constant>(910);
  auto gam_rho_tg = std::make_shared<Constant>(0.08);
  auto rate_rho = std::make_shared<Constant>(2.5);   
  auto gam_ts0 = std::make_shared<Constant>(0.4);
  auto gam_rho_mut = std::make_shared<Constant>(0.99);
  
  

  //Radio-chemo therapy effect on tumor cells density
  RT_TMZ_TC::BilinearForm a11(W8,W8);
  RT_TMZ_TC::LinearForm L11(W8);

  L11.w0=w0;
  L11.poro_b=poro_b0; L11.t_poro=poro0; 

  //Radio-chemo therapy effect on tumor cells necrosis
  RT_TMZ::BilinearForm a10(W7,W7);
  
  RT_TMZ::LinearForm L10(W7);
  L10.w0=w0; L10.wnecro_n=wnecro0;
  L10.poro_b=poro_b0; L10.t_poro=poro0; 
  L10.gam_tn=gam_tn;
  L10.poro_b_max=poro_b_max; L10.TMZ_t=TMZ_t;
  L10.rate_kill=rate_kill; L10.mMGMT=mMGMT;
  L10.a_=aECM; L10.sig_hl=sig_hl; L10.sig_th=sig_th;

  //ECM stiffening due to tumor cells phenotype
  Young::BilinearForm a9(W6,W6); 
  Young::LinearForm L9(W6);
  L9.w0=w0; L9.young_n=e_young;L9.w_crit0=w_crit0;
  L9.young_max=young_max;

  //Tumor cells necrosis due to O2 deprivation
  Necro::BilinearForm a5(W2,W2);  
  a5.w0=w0; a5.t_wrho=t_wrho;
  a5.dT=dt; a5.mu_t0=mu_t0; a5.a_=aECM; a5.k_max=k_max;
  a5.rho_s=rho_s; a5.rho_t=rho_t;
  a5.gam_rho_mut=gam_rho_mut; a5.gam_rho_tg=gam_rho_tg; 
  a5.w_crit0=w_crit0; a5.w_env=w_env; a5.p1=p1;
  a5.p_crit=p_crit; 
  a5.gam_ts0=gam_ts0; a5.sig_hl=sig_hl; a5.sig_th=sig_th;
  a5.e_permea=e_permea; a5.wnecro_n=wnecro0;
  a5.gam_tn=gam_tn; a5.gam_tg0=gam_tg0; a5.t_poro=poro0; 

  Necro::LinearForm L5(W2);
  L5.w0=w0;
  L5.dT=dt; L5.a_=aECM; 
  L5.rho_t=rho_t;
  L5.w_crit0=w_crit0;
  L5.sig_hl=sig_hl; L5.sig_th=sig_th;
  L5.wnecro_n=wnecro0;
  L5.t_poro=poro0; L5.gam_tn=gam_tn; 

  //Tumor cells phenotype switch due to mechano-biological coupling
  Rho::BilinearForm a7(W4,W4);
  
  Rho::LinearForm L7(W4);
  L7.w0=w0; L7.t_poro=poro0; L7.wrho_n=t_wrho;
  L7.wnecro=wnecro0; L7.w_crit0=w_crit0; L7.IDH_threshold=IDH_threshold;
  L7.sig_hl=sig_hl; L7.sig_th=sig_th;
  L7.a_=aECM; L7.rate_rho=rate_rho;

  //Regularization of numerical oscillation in necrotic tumor cells
  Reg_nec::BilinearForm a6(W3,W3);  
  Reg_nec::LinearForm L6(W3);
  L6.wnecro_n=wnecro0;L6.w0=w0;

  //Regularization of numerical oscillation in malignant tumor cells
  Reg_rho::BilinearForm a8(W5,W5);  
  Reg_rho::LinearForm L8(W5);
  L8.wrho_n=t_wrho;L8.w0=w0;

  //Porous system of Glioma
  Glioma::ResidualForm F(V); 
  F.w=w; F.w0=w0; F.t_wrho=t_wrho;
  F.t_poro=poro0; F.wb=poro_b0;F.diffusion=diffusion;
  F.dT=dt; F.mu_l=mu_l; F.mu_t0=mu_t0; F.mu_h=mu_h; F.a_=aECM; F.k_max=k_max; 
  F.rho_s=rho_s; F.rho_l=rho_l; F.rho_t=rho_t;
  F.gam_rho_mut=gam_rho_mut; F.gam_rho_tg=gam_rho_tg; 
  F.gam_b_nl=gam_b_nl;
  F.gam_nl_h=gam_nl_h; F.gam_nl_t=gam_nl_t;
  F.u=u_n;F.u_n=u_n;  
  F.gam_tg0=gam_tg0;F.w_crit_h=w_crit_h; 
  F.w_crit0=w_crit0; F.w_env=w_env; F.p1=p1; F.p_crit=p_crit;  
  F.gam_ts0=gam_ts0; F.sig_hl=sig_hl; F.sig_th=sig_th;
  F.e_permea=e_permea; F.wnecro=wnecro0; 
  F.D_0=D_0; F.delta=delta; 
 
  Glioma::JacobianForm J(V, V);
  J.w=w; J.w0=w0; J.t_wrho=t_wrho;
  J.t_poro=poro0; J.wb=poro_b0;J.diffusion=diffusion;
  J.dT=dt; J.mu_l=mu_l; J.mu_t0=mu_t0; J.mu_h=mu_h; J.a_=aECM; J.k_max=k_max;
  J.rho_s=rho_s; J.rho_l=rho_l; J.rho_t=rho_t;
  J.gam_rho_mut=gam_rho_mut; J.gam_rho_tg=gam_rho_tg;  
  J.gam_b_nl=gam_b_nl;
  J.gam_nl_h=gam_nl_h; J.gam_nl_t=gam_nl_t;
  J.u=u_n;J.u_n=u_n;
  J.gam_tg0=gam_tg0; J.w_crit_h=w_crit_h;
  J.w_crit0=w_crit0; J.w_env=w_env; J.p1=p1; J.p_crit=p_crit; 
  J.gam_ts0=gam_ts0; J.sig_hl=sig_hl; J.sig_th=sig_th;
  J.e_permea=e_permea; J.wnecro=wnecro0; 
  J.D_0=D_0; J.delta=delta;
 
  //Displacement field of the porous system
  Disp::BilinearForm a2(V2,V2);  
  a2.e_young=e_young;a2.nu=nu;a2.dT=dt;
    
  Disp::LinearForm L2(V2);
  L2.e_young=e_young;L2.nu=nu;L2.dT=dt;
  L2.u_n=u_n;
  L2.sig_hl=sig_hl; L2.sig_th=sig_th; L2.a_=aECM;
  L2.w0=w0;L2.w=w;

  //Evolution of the porosity of the system
  Poro::BilinearForm a3(V1,V1);
  a3.dT=dt;a3.u=u_n;a3.u_n=u_n;

  Poro::LinearForm L3(V1);
  L3.poro_n=poro0; L3.w=w; L3.w0=w0; L3.t_wrho=t_wrho;
  L3.rho_s=rho_s; L3.gam_ts0=gam_ts0; L3.gam_tg0=gam_tg0;
  L3.w_crit0=w_crit0; L3.w_env=w_env; L3.p1=p1;
  L3.p_crit=p_crit; L3.sig_hl=sig_hl; L3.sig_th=sig_th; L3.a_=aECM;
  L3.dT=dt; L3.wnecro=wnecro0;  
  L3.u=u_n;L3.u_n=u_n;
  
  //Regularization of numerical oscillation in tumor cells and O2
  Reg_w::BilinearForm a4(W1,W1);

  Reg_w::LinearForm L4(W1);
  L4.borne_pth=borne_pth; L4.borne_wnl=borne_wnl;
  L4.w0=w0;

  //// Create boundary conditions
  auto zero = std::make_shared<Constant>(0.0);
  auto bound = std::make_shared<on_bound>();
  
  DirichletBC bcpl(V->sub(0), zero, bound );
  DirichletBC bcphl(V->sub(1), zero, bound );
  DirichletBC bcpth(V->sub(2), zero, bound);
  DirichletBC bcwnl(V->sub(3), zero, bound);

  std::vector<const DirichletBC*> bcs = {{&bcpl,&bcphl,&bcpth,&bcwnl}};
  
  DirichletBC bcu1(V2->sub(0),zero, bound);
  DirichletBC bcu2(V2->sub(1),zero, bound);
  DirichletBC bcu3(V2->sub(2),zero, bound);
 
  std::vector<const DirichletBC*> bcdisp = {{&bcu1,&bcu2,&bcu3}};

  DirichletBC bcporo0(V1, zero_point_two,  bound);
  std::vector<const DirichletBC*> bcporo = {{&bcporo0}};

  DirichletBC bcpl_r(W1->sub(0), zero, bound );
  DirichletBC bcphl_r(W1->sub(1), zero, bound );
  DirichletBC bcpth_r(W1->sub(2), zero, bound);
  DirichletBC bcwnl_r(W1->sub(3), zero, bound);

  std::vector<const DirichletBC*> bcs_r = {{&bcpl_r,&bcphl_r,&bcpth_r,&bcwnl_r}};

  DirichletBC bcpl_RT(W8->sub(0), zero, bound );
  DirichletBC bcphl_RT(W8->sub(1), zero, bound );
  DirichletBC bcpth_RT(W8->sub(2), zero, bound);
  DirichletBC bcwnl_RT(W8->sub(3), zero, bound);
  std::vector<const DirichletBC*> bcs_RT = {{&bcpl_RT,&bcphl_RT,&bcpth_RT,&bcwnl_RT}};

  DirichletBC bcnec(W2, zero, bound);
  std::vector<const DirichletBC*> bcnecro = {{&bcnec}};
  
  DirichletBC bc_TMZ(W7, zero, bound);
  std::vector<const DirichletBC*> bcTMZ = {{&bc_TMZ}};

  DirichletBC bcnec_reg(W3, zero, bound);
  std::vector<const DirichletBC*> bcnecro_reg = {{&bcnec_reg}};
 
  DirichletBC bc_rho(W4, zero, bound);
  std::vector<const DirichletBC*> bcrho = {{&bc_rho}};
 
  DirichletBC bcwrho_reg(W5, zero, bound);
  std::vector<const DirichletBC*> bcrho_reg = {{&bcwrho_reg}};
  
  DirichletBC bc_young(W6, zero, bound);
  std::vector<const DirichletBC*> bcyoung = {{&bc_young}};
 
  //Solver parameters 
  Parameters params("nonlinear_variational_solver");
  Parameters newton_params("newton_solver");
  newton_params.add("relative_tolerance", 1e-15);
  newton_params.add("absolute_tolerance", 1e-10);
  newton_params.add("convergence_criterion", "incremental");
  newton_params.add("linear_solver", "mumps");
  params.add(newton_params);

  //Output file for visualization
  File u_file("Output/u.pvd"); 
  File pl_file("Output/pl.pvd");
  File phl_file("Output/phl.pvd");
  File pth_file("Output/pth.pvd");
  File wnl_file("Output/wnl.pvd");
  File poro_file("Output/poro.pvd");
  File necro_file("Output/necro.pvd");
  File rho_file("Output/rho.pvd");
  File young_file("Output/young.pvd");

  //Output file for calibration
  File X_update(mesh->mpi_comm(),"./To_update/X.xml.gz");
  File u_update(mesh->mpi_comm(),"./To_update/disp.xml.gz");
  File wrho_update(mesh->mpi_comm(),"./To_update/wrho.xml.gz");
  File poro_update(mesh->mpi_comm(),"./To_update/poro.xml.gz");
  File necro_update(mesh->mpi_comm(),"./To_update/necro.xml.gz");
  File young_update(mesh->mpi_comm(),"./To_update/young.xml.gz");

  pl_file << std::pair<const Function*, double>(&((*w0)[0]), t);
  phl_file << std::pair<const Function*, double>(&((*w0)[1]), t);
  pth_file << std::pair<const Function*, double>(&((*w0)[2]), t);
  wnl_file << std::pair<const Function*, double>(&((*w0)[3]), t);
  poro_file << std::pair<const Function*, double>(&(*poro0), t);
  u_file << std::pair<const Function*, double>(&(*u_n), t);      
  necro_file << std::pair<const Function*, double>(&(*wnecro0), t);
  rho_file << std::pair<const Function*, double>(&(*t_wrho), t);
  young_file << std::pair<const Function*, double>(&(*e_young), t);

  //Function for updating increment
  Function a(V); 
  Function a_poro(V1);
  Function a_u(V2); 
  Function a_young(W6); 

  //Increment of time for the solver (depends on glioma status, here T0)
  for (int N = 0; N < num_steps; N++)
  {
	      if (N == 27)
	      {	  *dt=60.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }
	      if (N == 47)
	      {	  *dt=120.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt;}
	      if (N == 67)
	      {	  *dt=240.0; F.dT=dt;J.dT=dt; a2.dT=dt; L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }
	      if (N == 77)
	      {	  *dt=480.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }
	      if (N == 87)
	      {	  *dt=600.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }
	      if (N == 97)
	      {	  *dt=900.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }
	      if (N == 107)
	      {	  *dt=1200.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }
	      if (N == 127)
	      {	  *dt=1800.0; F.dT=dt;J.dT=dt; a2.dT=dt;L2.dT=dt;  a3.dT=dt;L3.dT=dt;a5.dT=dt; L5.dT=dt; }


    t += *dt;
    cout << "Time: " << t <<  "sec   Iteration: " << N << endl;

  //Solve UFLs
  try{    
    solve(F == 0, *w, bcs, J,params);  
	L3.w=w;L2.w0=w;
  }
  catch (std::runtime_error) {
  
    	cout << "Solving PDE system failed " << endl;
  } 
  		
  cout << "Update porosity " << endl;
  try{    
	solve(a3 == L3, *poro, bcporo);	

	*poro0->vector() = *poro->vector();
	F.t_poro=poro0;J.t_poro=poro0;
	L3.poro_n=poro0;a5.t_poro=poro0; L5.t_poro=poro0; L7.t_poro=poro0;L10.t_poro=poro0;L11.t_poro=poro0;
	}

  catch (std::runtime_error) {

    	cout << "Solving porosity  failed " << endl;
	}

  cout << "Correct none physical values " << endl;
   	
  update_solution(a,*w,*w0);
  *w0=a;
  a.vector()->zero();
	
  try{    
	solve(a4 == L4, *w_reg,bcs_r);
	update_solution(a,*w_reg,*w0);
	*w0=a;	

	F.w0=w0; J.w0=w0;L7.w0=w0; L3.w0=w0; L2.w0=w0; L4.w0=w0;a5.w0=w0; L5.w0=w0; L6.w0=w0; L9.w0=w0;L10.w0=w0;L11.w0=w0;
  }	
  catch (std::runtime_error) {
   
    	cout << "Regularization failed " << endl;

  }
  
  cout << "Update displacement" << endl;
  F.u_n=u_n; J.u_n=u_n; a3.u_n=u_n;   
		
  try{
	solve(a2 == L2, *u, bcdisp);	    
	*u_n=*u;
	F.u=u_n; J.u=u_n; a3.u=u_n;
	L2.u_n=u_n;       
  }
  catch (std::runtime_error) {
  
    	cout << "Update displacement failed " << endl;
  }

 
    	cout << "Update necrotic tissue " << endl;
  try{
		solve(a5==L5,*wnecro,bcnecro);
		*wnecro0=*wnecro;
		L6.wnecro_n=wnecro0;
  }
  
  catch (std::runtime_error) {
  
    	cout << "Update necrosis failed " << endl;
  }

	cout << "Correct none physical values " << endl;
  try{
		solve(a6==L6,*wnecro_reg,bcnecro_reg);
		*wnecro0=*wnecro_reg;
		L7.wnecro=wnecro0;
		a5.wnecro_n=wnecro0; L5.wnecro_n=wnecro0; 
		F.wnecro=wnecro0; J.wnecro=wnecro0; 
		L3.wnecro=wnecro0; L10.wnecro_n=wnecro0;
  }
  catch (std::runtime_error) {
    	cout << "Regularization failed " << endl;
  }

		
  if (N%10==0){
    	cout << "Update Young modulus " << endl;		
		try{    
			solve(a9 == L9, *d_young,bcyoung);
			update_young(a_young,*d_young,*e_young);
			*e_young=a_young;	
			L9.young_n=e_young;
			a2.e_young=e_young; L2.e_young=e_young; 

		}	
		catch (std::runtime_error) {
   
				cout << "Update Young modulus failed " << endl;

		}
  }
  	

  if (N%48==0){   
	     cout << "RT_TMZ treatment " << endl;		
		try{    
			solve(a11 == L11, *w_RT,bcs_RT);
  			a.vector()->zero();
			update_solution(a,*w_RT,*w0);
			*w0=a;
			F.w0=w0; J.w0=w0;L7.w0=w0; L3.w0=w0; L2.w0=w0; L4.w0=w0;a5.w0=w0; L5.w0=w0; L6.w0=w0; L9.w0=w0;L10.w0=w0;L11.w0=w0;
		}	
		catch (std::runtime_error) {
   
				cout << "RT_TMZ treatment failed " << endl;

		}
	}

	//Output for visualization
	if (N%20==0){
		pl_file << std::pair<const Function*, double>(&((*w0)[0]), t);
		phl_file << std::pair<const Function*, double>(&((*w0)[1]), t);
		pth_file << std::pair<const Function*, double>(&((*w0)[2]), t);
		wnl_file << std::pair<const Function*, double>(&((*w0)[3]), t);
		poro_file << std::pair<const Function*, double>(&(*poro0), t);
		u_file << std::pair<const Function*, double>(&(*u_n), t);		
		necro_file << std::pair<const Function*, double>(&(*wnecro0), t);
		rho_file << std::pair<const Function*, double>(&(*t_wrho), t);
		young_file << std::pair<const Function*, double>(&(*e_young), t);
	}

	//Output for calibration
	if (N==1008){
		X_update << *w0;
		u_update << *u_n;
		wrho_update << *t_wrho;
		poro_update << *poro0;
		necro_update << *wnecro0;
		young_update << *e_young;
	}	
	w->vector()->zero();
	w->update();
	w_RT->vector()->zero();
	w_RT->update();
	u->update();
	wnecro->update();
	necro_RT->update();
	wrho->update();
	poro->update();
	d_young->vector()->zero();
	d_young->update();
	w_reg->update();
	wrho_reg->update();
	wnecro_reg->update();
	a.vector()->zero();
	a_young.vector()->zero();
	a_u.vector()->zero();
  }


  //Show all timings
  list_timings(TimingClear::clear, { TimingType::wall });

  return 0;
}
