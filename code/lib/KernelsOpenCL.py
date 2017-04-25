#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:21:40 2017

@author: diegothomas
"""
Kernel_Test = """
__kernel void Test(__global float *TSDF) {

        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        int z = get_global_id(2); /*depth*/
        TSDF[x + 512*y + 512*512*z] = 1.0f;
}
"""
#__global float *prevTSDF, __global float *Weight
#__read_only image2d_t VMap
Kernel_FuseTSDF = """
__kernel void FuseTSDF(__global float *TSDF,  __global float *Depth, __constant float *Param, __constant int *Dim,
                           __constant float *Pose, __constant float *calib, const int n_row, const int m_col) {
        //const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
        const float nu = 0.1;
            
        float4 pt;
        float4 pt_T;
        int2 pix;

        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        pt.x = ((float)(x)-Param[0])/Param[1];
        pt.y = ((float)(y)-Param[2])/Param[3];
        float x_T =  Pose[0]*pt.x + Pose[1]*pt.y + Pose[3];
        float y_T =  Pose[4]*pt.x + Pose[5]*pt.y + Pose[7];
        float z_T =  Pose[8]*pt.x + Pose[9]*pt.y + Pose[11];
             
        
        //Global computation

        //float Wlim = 10;
        
        for (int z = 0; z < 512; z++) { /*depth*/
            // Transform voxel coordinates into 3D point coordinates
            // Param = [c_x, dim_x, c_y, dim_y, c_z, dim_z]
            pt.z = ((float)(z)-Param[4])/Param[5];
            
            // Transfom the voxel into the Image coordinate space
            pt_T.x = x_T + Pose[2]*pt.z; //Pose is column major
            pt_T.y = y_T + Pose[6]*pt.z;
            pt_T.z = z_T + Pose[10]*pt.z;
            
            /* from here there is the copy for Diego's device and Inoe's device. */
            
            /*************************** Inoe **************************************/
            // Project onto Image
            pix.x = convert_int(round((pt_T.x/fabs(pt_T.z))*calib[0] + calib[2])); 
            pix.y = convert_int(round((pt_T.y/fabs(pt_T.z))*calib[4] + calib[5])); 
            
            if (pix.x < 0 || pix.x > m_col-1 || pix.y < 0 || pix.y > n_row-1) {
                TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = 1.0f;
                continue;
            }
            
            //float4 Proj_pt = read_imagef(VMap, smp, (int2){pix.x, pix.y});
            float dist = (pt_T.z - Depth[pix.x + m_col*pix.y]);
            
            if (dist > -nu)
               TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = min(1.0f, dist/nu);
            else
               TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = max(1.0f, dist/nu);//NULL;//
            
            
            
            
            // Global update
            //int idx = z + Dim[0]*y + Dim[0]*Dim[1]*x;
            //TSDF[idx] = (prevTSDF[idx]*Weight[idx] + TSDF[idx])/(1+Weight[idx]);
            
            //if (Weight[idx]+1 > Wlim) Weight[idx] = Wlim;
            //else Weight[idx] = Weight[idx]+1;
            
            /*************************** Diego **************************************/
            // Project onto Image
            /*
            pix.x = convert_int(round((pt_T[0]/fabs(pt_T[2]))*calib[0] + calib[2])); 
            pix.y = convert_int(round((pt_T[1]/fabs(pt_T[2]))*calib[4] + calib[5])); 
            
            
            if (pix.x < 0 || pix.x > m_col-1 || pix.y < 0 || pix.y > n_row-1) {
                TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = -1.0f;
                continue;
            }
            */    
            // float4 Proj_pt = read_imagef(VMap, smp, (int2){pix.x, pix.y});
            /*
            TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = (pt_T.z - Depth[pix.x + m_col*pix.y])/nu;
            */
            //if (dist > 0.0)
            //   TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = min(1.0f, dist/nu);
            //else
            //    TSDF[z + Dim[0]*y + Dim[0]*Dim[1]*x] = max(-1.0f, dist/nu);
        }
        
}
"""

Kernel_RayTracing = """
__kernel void RayTracing(__global float *TSDF, __global float *Depth, __constant float *Param, __constant int *Dim,
                           __constant float *Pose, __constant float *calib, const int n_row, const int m_col) {
        //const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
        float nu = 0.08f;

        int i = get_global_id(0); /*height*/
        int j = get_global_id(1); /*width*/
        
        // Shoot a ray for the current pixel
        float x = ((float)(j) - calib[2])/calib[0];
        float y = ((float)(i) - calib[5])/calib[4];
        
        float4 ray = {0.0f, 0.0f, 0.0f, 1.0f};
        ray.x = Pose[0]*x + Pose[1]*y + Pose[2]*1.0f + Pose[3]; //Pose is column major
        ray.y = Pose[4]*x + Pose[5]*y + Pose[6]*1.0f + Pose[7];
        ray.z = Pose[8]*x + Pose[9]*y + Pose[10]*1.0f + Pose[11];
        float norm = sqrt(ray.x*ray.x + ray.y*ray.y + ray.z*ray.z);
        ray.x = ray.x / norm; 
        ray.y = ray.y / norm; 
        ray.z = ray.z / norm;
        
        float4 pt = {0.5f*ray.x, 0.5f*ray.y, 0.5f*ray.z, 1.0f};
        int4 voxel = {convert_int(round(pt.x*Param[1] + Param[0])),
                      convert_int(round(pt.y*Param[3] + Param[2])),
                      convert_int(round(pt.z*Param[5] + Param[4])), 1};
        
        if (voxel.x < 0 || voxel.x > Dim[0]-1 || voxel.y < 0 || voxel.y > Dim[1]-1 || voxel.z< 0 || voxel.z > Dim[2]-1)
            return;
        
        float prev_TSDF = TSDF[voxel.z + Dim[0]*voxel.y + Dim[0]*Dim[1]*voxel.x];
        pt.x = pt.x + nu*ray.x;
        pt.y = pt.y + nu*ray.y;
        pt.z = pt.z + nu*ray.z;
        norm = sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
        
        float new_TSDF = 0.0f;
        float prev_norm = 0.0f;
        while (norm < 5.0f && prev_TSDF < 0.0f) {
            voxel.x = convert_int(round(pt.x*Param[1] + Param[0]));
            voxel.y = convert_int(round(pt.y*Param[3] + Param[2]));
            voxel.z = convert_int(round(pt.z*Param[5] + Param[4]));
            
            if (voxel.x < 0 || voxel.x > Dim[0]-1 || voxel.y < 0 || voxel.y > Dim[1]-1 || voxel.z < 0 || voxel.z > Dim[2]-1)
                return;
                
            new_TSDF = TSDF[voxel.z + Dim[0]*voxel.y + Dim[0]*Dim[1]*voxel.x];
                
            if (prev_TSDF*new_TSDF < 0.0f && prev_TSDF > -1.0f){
                Depth[j + m_col*i] = ((1.0f-fabs(prev_TSDF))*prev_norm + (1.0f-new_TSDF)*norm) / 
                                    (2.0f - (fabs(prev_TSDF) + new_TSDF));
                return;
            }
            
            if (new_TSDF > -1.0 && new_TSDF < 0.0)
                nu = 0.005f;
                
            prev_TSDF = new_TSDF;
            prev_norm = norm;
            pt.x = pt.x + nu*ray.x;
            pt.y = pt.y + nu*ray.y;
            pt.z = pt.z + nu*ray.z;
            norm = sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
        }
            
        Depth[j + m_col*i] = 0.0f;
}
"""