#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:37:19 2017

@author: diegothomas
"""

Kernel_TSDF = """
    int x = i/(dim_z*dim_y); 
    int y = (i%(dim_z*dim_y))/dim_z; 
    int z = (i%(dim_z*dim_y)) % dim_z;
    float3 pt = (float3) { ((float)(x)-Param[0])/Param[1],
                              ((float)(y)-Param[2])/Param[3],
                              ((float)(z)-Param[4])/Param[5] };
    float3 pt_T = (float3) {Pose[0]*pt.x + Pose[1]*pt.y + Pose[2]*pt.z + Pose[3],
                          Pose[4]*pt.x + Pose[5]*pt.y + Pose[6]*pt.z + Pose[7],
                          Pose[8]*pt.x + Pose[9]*pt.y + Pose[10]*pt.z + Pose[11]};
    // Project onto Image
    int2 pix = (int2){convert_int(round(Intrinsic[0]*pt_T.x/fabs(pt_T.z) + Intrinsic[2])),
                       convert_int(round(Intrinsic[1]*pt_T.y/fabs(pt_T.z) + Intrinsic[3]))};
    
    if (pix.x > -1 && pix.x < nbColumns && pix.y > -1 && pix.y < nbLines ) {
        if (depth[pix.x + nbColumns*pix.y] > 0.0f) {
            float dist = -(pt_T.z - depth[pix.x + nbColumns*pix.y])/nu;
            if (dist >= 1.0f) {
                Weight[i] = max(0, Weight[i]-1);
                if (Weight[i] == 0)
                    TSDF[i] = 30000;
            } else if (dist > -1.0f) {
                float prev_tsdf = convert_float(TSDF[i])/30000.0f;
                float new_tsdf = max(-1.0f, min(1.0f, (dist + convert_float(Weight[i])*prev_tsdf)/(convert_float(Weight[i]) + 1.0f)));
                // Global update
                TSDF[i] = convert_short(new_tsdf*30000.0f);
                Weight[i] = min(1000, Weight[i] + 1);
            } else if (Weight[i] == 0) {
                TSDF[i] = 30000;
            }
        } else if (Weight[i] == 0) {
            TSDF[i] = 30000;
        }
    } else if (Weight[i] == 0) {
        TSDF[i] = 30000;
    }
"""