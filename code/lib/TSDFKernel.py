#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:49:20 2017

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
    int2 pix = (int2){convert_int(round(Intrinsic[0]*pt_T.y/fabs(pt_T.z) + Intrinsic[2])),
                       convert_int(round(Intrinsic[1]*pt_T.x/fabs(pt_T.z) + Intrinsic[3]))};
    if (pix.x > -1 && pix.x < nbColumns && pix.y > -1 && pix.y < nbLines ) {
            float dist = max(-1.0f, min(1.0f, (-(pt_T.z - depth[pix.x + nbColumns*pix.y])/nu)));
            // Global update
            TSDF[i] = (depth[pix.x + nbColumns*pix.y] > 0.0f) ? convert_short(dist*30000.0f) : 30000;
    } else {
        TSDF[i] = 30000;
    }
"""