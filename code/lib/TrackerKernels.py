#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:19:44 2017

@author: diegothomas
"""

Kernel_GICP = """
if (i%3 == 0) {
    // Transform current 3D position and normal with current transformation
    float3 pt = (float3) {vmap[i], vmap[i+1], vmap[i+2]};
    float3 nmle = (float3) {nmap[i], nmap[i+1], nmap[i+2]};
    
    if (pt.z > 0.0f /*&& (fabs(nmle.x)+fabs(nmle.y)+fabs(nmle.z) != 0.0f)*/){
        
        float3 pt_T = (float3) {pt.x*Pose[0] + pt.y*Pose[1] + pt.z*Pose[2] + Pose[3],
                                pt.x*Pose[4] + pt.y*Pose[5] + pt.z*Pose[6] + Pose[7], 
                                pt.x*Pose[8] + pt.y*Pose[9] + pt.z*Pose[10] + Pose[11]};
        float3 nmle_T = (float3) {nmle.x*Pose[0] + nmle.y*Pose[1] + nmle.z*Pose[2], 
                                  nmle.x*Pose[4] + nmle.y*Pose[5] + nmle.z*Pose[6],
                                  nmle.x*Pose[8] + nmle.y*Pose[9] + nmle.z*Pose[10]};
        
        float norme_nmle = nmle_T.x*nmle_T.x + nmle_T.y*nmle_T.y + nmle_T.z*nmle_T.z;
        if (norme_nmle > 0.0f) {
        
            // Project onto Image2
            // pix = (line index, column index)
            int2 pix = (int2){convert_int(round(Intrinsic[1]*pt_T.y/fabs(pt_T.z) + Intrinsic[3])), 
                                  convert_int(round(Intrinsic[0]*pt_T.x/fabs(pt_T.z) + Intrinsic[2]))};
            
            if (pix.x > -1 && pix.x < nbLines && pix.y > -1 && pix.y < nbColumns ) {
                int idx_match = 3*(pix.y + pix.x*nbColumns);
                // Compute distance betwn matches and btwn normals
                float3 match_vtx = (float3) {vmap2[idx_match], vmap2[idx_match+1], vmap2[idx_match+2]}; 
                float3 match_nmle = (float3) {nmap2[idx_match], nmap2[idx_match+1], nmap2[idx_match+2]}; 
                
                float distance_v = (pt_T.x - match_vtx.x)*(pt_T.x - match_vtx.x) 
                                + (pt_T.y - match_vtx.y)*(pt_T.y - match_vtx.y) 
                                + (pt_T.z - match_vtx.z)*(pt_T.z - match_vtx.z);
                
                float distance_n = (nmle_T.x - match_nmle.x)*(nmle_T.x - match_nmle.x) 
                                + (nmle_T.y - match_nmle.y)*(nmle_T.y - match_nmle.y) 
                                + (nmle_T.z - match_nmle.z)*(nmle_T.z - match_nmle.z);
                
                if (distance_v < thresh_dis && distance_n < thresh_norm) {    
                    float w = 1.0f;
                    // Complete Jacobian matrix
                    Buffer_1[i/3] = w*nmle_T.x;
                    Buffer_2[i/3] = w*nmle_T.y;
                    Buffer_3[i/3] = w*nmle_T.z;
                    Buffer_4[i/3] = w*(-match_vtx.z*nmle_T.y + match_vtx.y*nmle_T.z);
                    Buffer_5[i/3] = w*(match_vtx.z*nmle_T.x - match_vtx.x*nmle_T.z);
                    Buffer_6[i/3] = w*(-match_vtx.y*nmle_T.x + match_vtx.x*nmle_T.y);
                    Buffer_B[i/3] = w*(nmle_T.x*(match_vtx.x - pt_T.x) + 
                               nmle_T.y*(match_vtx.y - pt_T.y) + 
                               nmle_T.z*(match_vtx.z - pt_T.z));
                } else {
                    // buffer = 0
                    Buffer_1[i/3] = 0.f;
                    Buffer_2[i/3] = 0.f;
                    Buffer_3[i/3] = 0.f;
                    Buffer_4[i/3] = 0.f;
                    Buffer_5[i/3] = 0.f;
                    Buffer_6[i/3] = 0.f;
                    Buffer_B[i/3] = 0.f;
                }
            } else {
                //buffer = 0
                Buffer_1[i/3] = 0.f;
                Buffer_2[i/3] = 0.f;
                Buffer_3[i/3] = 0.f;
                Buffer_4[i/3] = 0.f;
                Buffer_5[i/3] = 0.f;
                Buffer_6[i/3] = 0.f;
                Buffer_B[i/3] = 0.f;
            }
        } else {
            // buffer =0
            Buffer_1[i/3] = 0.f;
            Buffer_2[i/3] = 0.f;
            Buffer_3[i/3] = 0.f;
            Buffer_4[i/3] = 0.f;
            Buffer_5[i/3] = 0.f;
            Buffer_6[i/3] = 0.f;
            Buffer_B[i/3] = 0.f;
        }
    } else {
        Buffer_1[i/3] = 0.f;
        Buffer_2[i/3] = 0.f;
        Buffer_3[i/3] = 0.f;
        Buffer_4[i/3] = 0.f;
        Buffer_5[i/3] = 0.f;
        Buffer_6[i/3] = 0.f;
        Buffer_B[i/3] = 0.f;
    }
}                
"""

KERNEL_DOT = """
// Thread block size
#define BLOCK_SIZE 16
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA 50*16 // Matrix A width
#define HA 100*16 // Matrix A height
#define WB 50*16 // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height
/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */
/* Matrix multiplication: C = A * B.
 * Device code.
 */
#define AS(j, i) As[i + j * BLOCK_SIZE]
#define BS(j, i) Bs[i + j * BLOCK_SIZE]
////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1))) 
void
My_dot( __global float* C, __global float* A, __global float* B)
{
    if (get_global_id(0) > 5 || get_global_id(1) > 5)
        return;
        
    __local float As[BLOCK_SIZE*BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE*BLOCK_SIZE];
    
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
        
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WA - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WB;
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + WA * ty + tx];  // Transpose matrix A
        BS(ty, tx) = B[b + WB * ty + tx];
        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * 6 + get_global_id(0)] = Csub;
}
"""