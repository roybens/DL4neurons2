#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _Ca_reg(void);
extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _KdShu2007_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _ProbAMPANMDA_EMS_reg(void);
extern void _ProbGABAAB_EMS_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);
extern void _StochKv_reg(void);
extern void _cad_reg(void);
extern void _izhi2003a_reg(void);
extern void _kca_reg(void);
extern void _km_reg(void);
extern void _kv_reg(void);
extern void _na_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," modfiles/Ca.mod");
    fprintf(stderr," modfiles/CaDynamics_E2.mod");
    fprintf(stderr," modfiles/Ca_HVA.mod");
    fprintf(stderr," modfiles/Ca_LVAst.mod");
    fprintf(stderr," modfiles/Ih.mod");
    fprintf(stderr," modfiles/Im.mod");
    fprintf(stderr," modfiles/K_Pst.mod");
    fprintf(stderr," modfiles/K_Tst.mod");
    fprintf(stderr," modfiles/KdShu2007.mod");
    fprintf(stderr," modfiles/NaTa_t.mod");
    fprintf(stderr," modfiles/NaTs2_t.mod");
    fprintf(stderr," modfiles/Nap_Et2.mod");
    fprintf(stderr," modfiles/ProbAMPANMDA_EMS.mod");
    fprintf(stderr," modfiles/ProbGABAAB_EMS.mod");
    fprintf(stderr," modfiles/SK_E2.mod");
    fprintf(stderr," modfiles/SKv3_1.mod");
    fprintf(stderr," modfiles/StochKv.mod");
    fprintf(stderr," modfiles/cad.mod");
    fprintf(stderr," modfiles/izhi2003a.mod");
    fprintf(stderr," modfiles/kca.mod");
    fprintf(stderr," modfiles/km.mod");
    fprintf(stderr," modfiles/kv.mod");
    fprintf(stderr," modfiles/na.mod");
    fprintf(stderr, "\n");
  }
  _Ca_reg();
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _KdShu2007_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _ProbAMPANMDA_EMS_reg();
  _ProbGABAAB_EMS_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
  _StochKv_reg();
  _cad_reg();
  _izhi2003a_reg();
  _kca_reg();
  _km_reg();
  _kv_reg();
  _na_reg();
}
