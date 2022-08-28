#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CaDynamics_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVA_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _Im_v2_reg(void);
extern void _K_P_reg(void);
extern void _K_T_reg(void);
extern void _Kd_reg(void);
extern void _KdT_reg(void);
extern void _Kv2like_reg(void);
extern void _Kv3_1_reg(void);
extern void _Kv3_1T_reg(void);
extern void _NaTa_reg(void);
extern void _NaTs_reg(void);
extern void _NaV_reg(void);
extern void _NaV2_reg(void);
extern void _Nap_reg(void);
extern void _SK_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"allen/modfiles/CaDynamics.mod\"");
    fprintf(stderr," \"allen/modfiles/Ca_HVA.mod\"");
    fprintf(stderr," \"allen/modfiles/Ca_LVA.mod\"");
    fprintf(stderr," \"allen/modfiles/Ih.mod\"");
    fprintf(stderr," \"allen/modfiles/Im.mod\"");
    fprintf(stderr," \"allen/modfiles/Im_v2.mod\"");
    fprintf(stderr," \"allen/modfiles/K_P.mod\"");
    fprintf(stderr," \"allen/modfiles/K_T.mod\"");
    fprintf(stderr," \"allen/modfiles/Kd.mod\"");
    fprintf(stderr," \"allen/modfiles/KdT.mod\"");
    fprintf(stderr," \"allen/modfiles/Kv2like.mod\"");
    fprintf(stderr," \"allen/modfiles/Kv3_1.mod\"");
    fprintf(stderr," \"allen/modfiles/Kv3_1T.mod\"");
    fprintf(stderr," \"allen/modfiles/NaTa.mod\"");
    fprintf(stderr," \"allen/modfiles/NaTs.mod\"");
    fprintf(stderr," \"allen/modfiles/NaV.mod\"");
    fprintf(stderr," \"allen/modfiles/NaV2.mod\"");
    fprintf(stderr," \"allen/modfiles/Nap.mod\"");
    fprintf(stderr," \"allen/modfiles/SK.mod\"");
    fprintf(stderr, "\n");
  }
  _CaDynamics_reg();
  _Ca_HVA_reg();
  _Ca_LVA_reg();
  _Ih_reg();
  _Im_reg();
  _Im_v2_reg();
  _K_P_reg();
  _K_T_reg();
  _Kd_reg();
  _KdT_reg();
  _Kv2like_reg();
  _Kv3_1_reg();
  _Kv3_1T_reg();
  _NaTa_reg();
  _NaTs_reg();
  _NaV_reg();
  _NaV2_reg();
  _Nap_reg();
  _SK_reg();
}
