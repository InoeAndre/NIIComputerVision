#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:45:44 2017

@author: diegothomas
"""

Preambule_MC = """
//    create the precalculated 256 possible polygon configuration (128 + symmetries)

__constant int Config[128][4][3] = { {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}}, //0
          {{0, 7, 3}, {0,0,0}, {0,0,0}, {0,0,0}}, // v0 1
          {{0, 1, 4}, {0,0,0}, {0,0,0}, {0,0,0}}, // v1 2
          {{1, 7, 3}, {1, 4, 7}, {0,0,0}, {0,0,0}}, // v0|v1 3
          {{1, 2, 5}, {0,0,0}, {0,0,0}, {0,0,0}}, // v2 4
          {{0, 7, 3}, {1, 2, 5}, {0,0,0}, {0,0,0}}, //v0|v2 5
          {{0, 2 ,4}, {2, 5, 4}, {0,0,0}, {0,0,0}}, //v1|v2 6
          {{3, 2, 7}, {2, 5, 7}, {5, 4, 7}, {0,0,0}}, //v0|v1|v2 7
          {{2, 3, 6}, {0,0,0}, {0,0,0}, {0,0,0}}, // v3 8
          {{0, 7, 2}, {7, 6, 2}, {0,0,0}, {0,0,0}}, //v0|v3 9
          {{0, 1, 4}, {3, 6, 2}, {0,0,0}, {0,0,0}}, //v1|v3 10
          {{1, 4, 2}, {2, 4, 6}, {4, 7, 6}, {0,0,0}}, //v0|v1|v3 11
          {{3, 5, 1}, {3, 6, 5}, {0,0,0}, {0,0,0}}, //v2|v3 12
          {{0, 7, 1}, {1, 7, 5}, {7, 6, 5}, {0,0,0}}, //v0|v2|v3 13
          {{0, 3, 4}, {3, 6, 4}, {6, 5, 4}, {0,0,0}}, //v1|v2|v3 14
          {{7, 6, 5}, {4, 7, 5}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3 15
          {{7, 8, 11}, {0,0,0}, {0,0,0}, {0,0,0}}, // v4 16
          {{0, 8, 3}, {3, 8, 11}, {0,0,0}, {0,0,0}}, //v0|v4 17
          {{0, 1, 4}, {7, 8, 11}, {0,0,0}, {0,0,0}}, //v1|v4 18
          {{1, 4, 8}, {1, 8, 11}, {1, 11, 3}, {0,0,0}}, //v0|v1|v4 19
          {{7, 8, 11}, {1, 2, 5}, {0,0,0}, {0,0,0}}, //v2|v4 20
          {{0, 8, 3}, {3, 8, 11}, {1, 2, 5}, {0,0,0}}, //v0|v2|v4 21
          {{0, 2 ,4}, {2, 5, 4}, {7, 8, 11}, {0,0,0}}, //v1|v2|v4 22
          {{4, 2, 5}, {2, 4, 11}, {4, 8, 11}, {2, 11, 3}}, //v0|v1|v2|v4 23
          {{2, 3, 6}, {7, 8, 11}, {0,0,0}, {0,0,0}}, //v3|v4 24
          {{6, 8, 11}, {2, 8, 6}, {0, 8, 2}, {0,0,0}}, //v0|v3|v4 25
          {{0, 1, 4}, {2, 3, 6}, {7, 8, 11}, {0,0,0}}, //v1|v3|v4 26
          {{6, 8, 11}, {8, 6, 2}, {4, 8, 2}, {4, 2, 1}}, //v0|v1|v3|v4 27
          {{7, 8, 11}, {3, 5, 1}, {3, 6, 5}, {0,0,0}}, //v2|v3|v4 28
          {{0, 8 ,1}, {1, 8, 6}, {1, 6, 5}, {6, 8, 11}}, //v0|v2|v3|v4 29
          {{7, 8, 11}, {0, 3, 4}, {3, 6, 4}, {6, 5, 4}}, //v1|v2|v3|v4 30
          {{6, 8, 11}, {6, 4, 8}, {5, 4, 6}, {0,0,0}}, //v0|v1|v2|v3|v4 31 =                              v5|v6|v7   //////////////////////////
          {{4, 9, 8}, {0,0,0}, {0,0,0}, {0,0,0}}, // v5 32
          {{0, 7, 3}, {4, 9, 8}, {0,0,0}, {0,0,0}}, //v0|v5 33
          {{0, 1, 8}, {1, 9, 8}, {0,0,0}, {0,0,0}}, //v1|v5 34
          {{1, 9, 3}, {3, 9, 7}, {7, 9, 8}, {0,0,0}}, //v0|v1|v5 35
          {{4, 9, 8}, {1, 2, 5}, {0,0,0}, {0,0,0}}, //v2|v5 36
          {{4, 9, 8}, {1, 2, 5}, {0, 7, 3}, {0,0,0}}, //v0|v2|v5 37
          {{0, 2 ,8}, {2, 5, 8}, {8, 5, 9}, {0,0,0}}, //v1|v2|v5 38
          {{7, 9, 8}, {3, 9, 7}, {3, 5, 9}, {2, 5, 3}}, //v0|v1|v2|v5 39
          {{4, 9, 8}, {2, 3, 6}, {0,0,0}, {0,0,0}}, //v3|v5 40
          {{4, 9, 8}, {0, 7, 2}, {7, 6, 2}, {0,0,0}}, //v0|v3|v5 41
          {{2, 3, 6}, {0, 1, 8}, {1, 9, 8}, {0,0,0}}, //v1|v3|v5 42
          {{1, 9, 2}, {2, 9, 7}, {7, 6, 9}, {7, 9, 8}}, //v0|v1|v3|v5 43
          {{4, 9, 8}, {3, 5, 1}, {3, 6, 5}, {0,0,0}}, //v2|v3|v5 44
          {{4, 9, 8}, {0, 7, 1}, {1, 7, 5}, {7, 6, 5}}, //v0|v2|v3|v5 45
          {{5, 9, 8}, {0, 3, 8}, {3, 5, 8}, {3, 6, 5}}, //v1|v2|v3|v5 46
          {{5, 7, 6}, {8, 5, 9}, {7, 5, 8}, {0,0,0}}, //v0|v1|v2|v3|v5 47                                     = v4 | v6 | v7  ////////////////////
          {{4, 9, 7}, {7, 9, 11}, {0,0,0}, {0,0,0}}, //v4|v5 48
          {{3, 9, 11}, {0, 4, 3}, {4, 9, 3}, {0,0,0}}, //v0|v4|v5 49
          {{1, 9, 11}, {0, 11, 7}, {0, 1, 11}, {0,0,0}}, //v1|v4|v5 50
          {{1, 9, 11}, {1, 11, 3}, {0,0,0}, {0,0,0}}, //v0|v1|v4|v5 51
          {{1, 2, 5}, {4, 9, 7}, {7, 9, 11}, {0,0,0}}, //v2|v4|v5 52
          {{1, 2, 5}, {3, 9, 11}, {0, 4, 3}, {4, 9, 3}}, //v0|v2|v4|v5 53
          {{0, 2, 7}, {2, 5, 9}, {2, 9, 7}, {7, 9, 11}}, //v1|v2|v4|v5 54
          {{11, 3, 9}, {3, 2, 5}, {9, 3, 5}, {0,0,0}}, //v0|v1|v2|v4|v5 55                                          = v3 v6 v7 //////////////////////////
          {{2, 3, 6}, {4, 9, 7}, {7, 9, 11}, {0,0,0}}, //v3|v4|v5 56
          {{2, 0, 4}, {6, 2, 11}, {2, 4, 11}, {4, 9, 11}}, //v0|v3|v4|v5 57
          {{2, 3, 6}, {1, 2, 5}, {4, 9, 7}, {7, 9, 11}}, //v1|v3|v4|v5 58
          {{11, 1, 9}, {2, 1, 6}, {6, 1, 11}, {0,0,0}}, //v0|v1|v3|v4|v5 59                                                               = v2 v6 v7 //////////////////
          {{7, 4, 11}, {4, 9, 11}, {3, 6, 1}, {1, 6, 5}}, //v2|v3|v4|v5 60
          {{1, 0, 4}, {11, 6, 9}, {9, 6, 5}, {0,0,0}}, //v0|v2|v3|v4|v5 61                                                                  = v1 v6 v7
          {{3, 0, 7}, {11, 6, 9}, {9, 6, 5}, {0,0,0}}, //v1|v2|v3|v4|v5 62                                                                 = v0 v6 v7
          {{11, 6, 9}, {9, 6, 5}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3|v4|v5 63                                                                 = v6 v7
          {{5, 10, 9}, {0,0,0}, {0,0,0}, {0,0,0}}, //v6 64
          {{5, 10, 9}, {0, 7, 3}, {0,0,0}, {0,0,0}}, //v0|v6 65
          {{5, 10, 9}, {0, 1, 4}, {0,0,0}, {0,0,0}}, //v1|v6 66
          {{5, 10, 9}, {1, 3, 7}, {1, 7, 4}, {0,0,0}}, //v0|v1|v6 67
          {{1, 2, 9}, {2, 10, 9}, {0,0,0}, {0,0,0}}, //v2|v6 68
          {{1, 2, 9}, {2, 10, 9}, {0, 7, 3}, {0,0,0}}, //v0|v2|v6 69
          {{0, 2, 10}, {4, 10, 9}, {0, 10, 4}, {0,0,0}}, //v1|v2|v6 70
          {{2, 10, 3}, {4, 10, 9}, {4, 3, 10}, {3, 4, 7}}, //v0|v1|v2|v6 71
          {{5, 10, 9}, {2, 3, 6}, {0,0,0}, {0,0,0}}, //v3|v6 72
          {{5, 10, 9}, {0, 7, 2}, {7, 6, 2}, {0,0,0}}, //v0|v3|v6 73
          {{5, 10, 9}, {0, 1, 4}, {2, 3, 6}, {0,0,0}}, //v1|v3|v6 74
          {{5, 10, 9}, {1, 4, 2}, {2, 4, 6}, {4, 6, 7}}, //v0|v1|v3|v6 75
          {{1, 3, 9}, {6, 10, 9}, {3, 6, 9}, {0,0,0}}, //v2|v3|v6 76
          {{0, 7, 6}, {6, 10, 9}, {0, 6, 9}, {0, 9, 1}}, //v0|v2|v3|v6 77
          {{6, 10, 9}, {3, 6, 9}, {3, 9, 4}, {0, 3, 4}}, //v1|v2|v3|v6 78
          {{4, 7, 6}, {4, 10, 9}, {4, 6, 10}, {0,0,0}}, //v0|v1|v2|v3|v6 79    v4 v5 v7   ////////////////////////////
          {{5, 10, 9}, {7, 8, 11}, {0,0,0}, {0,0,0}}, //v4|v6 80
          {{5, 10, 9}, {0, 8, 3}, {3, 8, 11}, {0,0,0}}, //v0|v4|v6 81
          {{0, 1, 4}, {7, 8, 11}, {5, 10, 9}, {0,0,0}}, //v1|v4|v6 82
          {{5, 10, 9}, {1, 4, 8}, {1, 8, 11}, {1, 11, 3}}, //v0|v1|v4|v6 83
          {{1, 2, 9}, {2, 10, 9}, {7, 8, 11}, {0,0,0}}, //v2|v4|v6 84
          {{1, 2, 9}, {2, 10, 9}, {0, 8, 3}, {3, 8, 11}}, //v0|v2|v4|v6 85
          {{7, 8, 11}, {0, 2, 10}, {4, 10, 9}, {0, 10, 4}}, //v1|v2|v4|v6 86
          {{4, 8, 9}, {3, 2, 11}, {2, 10, 11}, {0,0,0}}, //v0|v1|v2|v4|v6 87                      = v3 v5 v7
          {{2, 3, 6}, {7, 8, 11}, {5, 10, 9}, {0,0,0}}, //v3|v4|v6 88
          {{5, 10, 9}, {6, 8, 11}, {2, 8, 6}, {0, 8, 2}}, //v0|v3|v4|v6 89
          {{0, 1, 4}, {2, 3, 6}, {7, 8, 11}, {5, 10, 9}}, //v1|v3|v4|v6 90                                         
          {{2, 1, 5}, {9, 4, 8}, {11, 6, 10}, {0,0,0}}, //v0|v1|v3|v4|v6 91     = v2 v5 v7   //////////////////////
          {{7, 8, 11}, {1, 3, 9}, {6, 10, 9}, {3, 6, 9}}, //v2|v3|v4|v6 92
          {{11, 6, 10}, {1, 0, 8}, {9, 1, 8}, {0,0,0}}, //v0|v2|v3|v4|v6 93                              = v1 v5 v7 //////////
          {{0, 3, 7}, {9, 4, 8}, {10, 5, 9}, {0,0,0}}, //v1|v2|v3|v4|v6 94                                      = v0 v5 v7  ////////////
          {{11, 6, 10}, {9, 4, 8}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3|v4|v6 95                                               = v5 v7 ////////////
          {{4, 5, 8}, {8, 5, 10}, {0,0,0}, {0,0,0}}, //v5|v6 96
          {{0, 7, 3}, {4, 5, 8}, {8, 5, 10}, {0,0,0}}, //v0|v5|v6 97
          {{0, 10, 8}, {1, 5, 10}, {0, 1, 10}, {0,0,0}}, //v1|v5|v6 98
          {{1, 5, 10}, {8, 7, 10}, {1, 10, 7}, {3, 1, 7}}, //v0|v1|v5|v6 99
          {{2, 10, 8}, {1, 8, 4}, {1, 2, 8}, {0,0,0}}, //v2|v5|v6 100
          {{2, 10, 8}, {1, 8, 4}, {1, 2, 8}, {0, 7, 3}}, //v0|v2|v5|v6 101
          {{0, 10, 8}, {0, 2, 10}, {0,0,0}, {0,0,0}}, //v1|v2|v5|v6 102
          {{8, 2, 10}, {3, 8, 7}, {3, 2, 8}, {0,0,0}}, //v0|v1|v2|v5|v6 103                                                         = v3 v4 v7 //////////////
          {{2, 3, 6}, {4, 5, 8}, {8, 5, 10}, {0,0,0}}, //v3|v5|v6 104
          {{4, 5, 8}, {8, 5, 10}, {0, 7, 2}, {7, 6, 2}}, //v0|v3|v5|v6 105
          {{2, 3, 6}, {0, 10, 8}, {1, 5, 10}, {0, 1, 10}}, //v1|v3|v5|v6 106
          {{1, 5, 2}, {7, 6, 8}, {6, 10, 8}, {0,0,0}}, //v0|v1|v3|v5|v6 107                                                 =v2 v4 v7 ////////////////////////
          {{3, 6, 10}, {1, 3, 4}, {4, 3, 10}, {4, 10, 8}}, //v2|v3|v5|v6 108
          {{1, 0, 4}, {7, 6, 8}, {6, 10, 8}, {0,0,0}}, //v0|v2|v3|v5|v6 109                                                 = v1 v4 v7 //////////
          {{0, 10, 8}, {0, 3, 6}, {0, 6, 10}, {0,0,0}}, //v1|v2|v3|v5|v6 110                                                = v0 v4 v7 //////////
          {{7, 6, 8}, {6, 10, 8}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v3|v5|v6 111                                                = v4 v7 //////////
          {{4, 5, 7}, {7, 10, 11}, {7, 5, 10}, {0,0,0}}, //v4|v5|v6 112
          {{5, 10, 11}, {0, 4, 5}, {0, 5, 11}, {0, 11, 3}}, //v0|v4|v5|v6 113
          {{0, 1, 5}, {0, 5, 10}, {0, 10, 7}, {7, 10, 11}}, //v1|v4|v5|v6 114
          {{3, 1, 11}, {1, 5, 10}, {1, 10, 11}, {0,0,0}}, //v0|v1|v4|v5|v6 115                                             = v2 v3 v7 ////////////
          {{7, 10, 11}, {4, 11, 7}, {4, 10, 11}, {1, 10, 4}}, //v2|v4|v5|v6 116
          {{0, 4, 1}, {3, 2, 11}, {2, 10, 11}, {0,0,0}}, //v0|v2|v4|v5|v6 117                                            = v1 v3 v7 ////////////
          {{0, 2, 10}, {0, 11, 7}, {0, 10, 11}, {0,0,0}}, //v1|v2|v4|v5|v6 118                                            = v0 v3 v7 ////////////
          {{3, 2, 11}, {2, 10, 11}, {0,0,0}, {0,0,0}}, //v0|v1|v2|v4|v5|v6 119                                           = v3 v7 ////////////
          {{2, 3, 6}, {4, 5, 7}, {7, 10, 11}, {7, 5, 10}}, //v3|v4|v5|v6 120
          {{6, 10, 11}, {0, 4, 2}, {2, 4, 5}, {0,0,0}}, //v0|v3|v4|v5|v6 121                                          = v1 v2 v7 ////////////
          {{3, 0, 7}, {2, 1, 5}, {11, 6, 10}, {0,0,0}}, //v1|v3|v4|v5|v6 122                                          = v0 v2 v7 ////////////
          {{2, 1, 5}, {11, 6, 10}, {0,0,0}, {0,0,0}}, //v0|v1|v3|v4|v5|v6 123                                          = v2 v7 ////////////
          {{3, 1, 7}, {7, 1, 4}, {11, 6, 10}, {0,0,0}}, //v2|v3|v4|v5|v6 124                                         = v0 v1 v7 ////////////
          {{1, 0, 4}, {11, 6, 10}, {0,0,0}, {0,0,0}}, //v0|v2|v3|v4|v5|v6 125                                         = v1 v7 ////////////
          {{3, 0, 7}, {11, 6, 10}, {0,0,0}, {0,0,0}}, //v1|v2|v3|v4|v5|v6 126                                         = v0 v7 ////////////
          {{11, 6, 10}, {0,0,0}, {0,0,0}, {0,0,0}} //v0|v1|v2|v3|v4|v5|v6 127                                         = v7 ////////////
          };

__constant int ConfigCount[128] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 
2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 
3, 2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 
3, 2, 3, 2, 2, 1 };
"""

Kernel_MarchingCube = """
int x = i/(dim_z*dim_y); 
int y = (i%(dim_z*dim_y))/dim_z; 
int z = (i%(dim_z*dim_y)) % dim_z;

if (x < dim_x-1 && y < dim_y-1 && z < dim_z-1) {

    float s[8][3] = {{0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
         {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}};
    
    float v[12][3] = {{0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
         {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f},
         {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}, {0.0f,0.0f, 0.0f}};
    
    float vals[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    
    // get the 8  current summits
    s[0][0] = (convert_float(x) - Param[0])/Param[1];
    s[0][1] = (convert_float(y) - Param[2])/Param[3]; 
    s[1][0] = (convert_float(x+1) - Param[0])/Param[1]; 
    s[1][1] = (convert_float(y) - Param[2])/Param[3]; 
    s[2][0] = (convert_float(x+1) - Param[0])/Param[1]; 
    s[2][1] = (convert_float(y+1) - Param[2])/Param[3]; 
    s[3][0] = (convert_float(x) - Param[0])/Param[1]; 
    s[3][1] = (convert_float(y+1) - Param[2])/Param[3]; 
    s[4][0] = (convert_float(x) - Param[0])/Param[1]; 
    s[4][1] = (convert_float(y) - Param[2])/Param[3]; 
    s[5][0] = (convert_float(x+1) - Param[0])/Param[1]; 
    s[5][1] = (convert_float(y) - Param[2])/Param[3]; 
    s[6][0] = (convert_float(x+1) - Param[0])/Param[1]; 
    s[6][1] = (convert_float(y+1) - Param[2])/Param[3]; 
    s[7][0] = (convert_float(x) - Param[0])/Param[1]; 
    s[7][1] = (convert_float(y+1) - Param[2])/Param[3]; 
    
    s[0][2] = (convert_float(z) - Param[4])/Param[5];
    s[1][2] = (convert_float(z) - Param[4])/Param[5]; 
    s[2][2] = (convert_float(z) - Param[4])/Param[5]; 
    s[3][2] = (convert_float(z) - Param[4])/Param[5];
    s[4][2] = (convert_float(z+1) - Param[4])/Param[5];
    s[5][2] = (convert_float(z+1) - Param[4])/Param[5];
    s[6][2] = (convert_float(z+1) - Param[4])/Param[5];
    s[7][2] = (convert_float(z+1) - Param[4])/Param[5];
        
    bool reverse = false;
    
    // get the index value corresponding to the implicit function
    int index = convert_int(IndexVal[i]);
    
    if (index != 0) {
        
        reverse = false;
        if (index < 0) {
            reverse = true;
            index = -index;
        }
        
        int nb_faces = ConfigCount[index];
        if (nb_faces > 0) {
        
            // get the values of the implicit function at the summits
            // [val_0 ... val_7]
            vals[0] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z + dim_z*y + dim_z*dim_y*x])/30000.0f));
            vals[1] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z + dim_z*y + dim_z*dim_y*(x+1)])/30000.0f));
            vals[2] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z + dim_z*(y+1) + dim_z*dim_y*(x+1)])/30000.0f));
            vals[3] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z + dim_z*(y+1) + dim_z*dim_y*x])/30000.0f));
            vals[4] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z+1 + dim_z*y + dim_z*dim_y*x])/30000.0f));
            vals[5] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z+1 + dim_z*y + dim_z*dim_y*(x+1)])/30000.0f));
            vals[6] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z+1 + dim_z*(y+1) + dim_z*dim_y*(x+1)])/30000.0f));
            vals[7] = 1.0f/(0.00001f + fabs(convert_float(TSDF[z+1 + dim_z*(y+1) + dim_z*dim_y*x])/30000.0f));
        
            int offset = Offset[i];
            
            v[0][0] = (vals[0]*s[0][0] + vals[1]*s[1][0])/(vals[0]+vals[1]); v[0][1] = (vals[0]*s[0][1] + vals[1]*s[1][1])/(vals[0]+vals[1]);
            v[1][0] = (vals[1]*s[1][0] + vals[2]*s[2][0])/(vals[1]+vals[2]); v[1][1] = (vals[1]*s[1][1] + vals[2]*s[2][1])/(vals[1]+vals[2]);
            v[2][0] = (vals[2]*s[2][0] + vals[3]*s[3][0])/(vals[2]+vals[3]); v[2][1] = (vals[2]*s[2][1] + vals[3]*s[3][1])/(vals[2]+vals[3]); 
            v[3][0] = (vals[0]*s[0][0] + vals[3]*s[3][0])/(vals[0]+vals[3]); v[3][1] = (vals[0]*s[0][1] + vals[3]*s[3][1])/(vals[0]+vals[3]); 
            v[4][0] = (vals[1]*s[1][0] + vals[5]*s[5][0])/(vals[1]+vals[5]); v[4][1] = (vals[1]*s[1][1] + vals[5]*s[5][1])/(vals[1]+vals[5]); 
            v[5][0] = (vals[2]*s[2][0] + vals[6]*s[6][0])/(vals[2]+vals[6]); v[5][1] = (vals[2]*s[2][1] + vals[6]*s[6][1])/(vals[2]+vals[6]); 
            v[6][0] = (vals[3]*s[3][0] + vals[7]*s[7][0])/(vals[3]+vals[7]); v[6][1] = (vals[3]*s[3][1] + vals[7]*s[7][1])/(vals[3]+vals[7]); 
            v[7][0] = (vals[0]*s[0][0] + vals[4]*s[4][0])/(vals[0]+vals[4]); v[7][1] = (vals[0]*s[0][1] + vals[4]*s[4][1])/(vals[0]+vals[4]); 
            v[8][0] = (vals[4]*s[4][0] + vals[5]*s[5][0])/(vals[4]+vals[5]); v[8][1] = (vals[4]*s[4][1] + vals[5]*s[5][1])/(vals[4]+vals[5]); 
            v[9][0] = (vals[5]*s[5][0] + vals[6]*s[6][0])/(vals[5]+vals[6]); v[9][1] = (vals[5]*s[5][1] + vals[6]*s[6][1])/(vals[5]+vals[6]); 
            v[10][0] = (vals[6]*s[6][0] + vals[7]*s[7][0])/(vals[6]+vals[7]); v[10][1] = (vals[6]*s[6][1] + vals[7]*s[7][1])/(vals[6]+vals[7]);
            v[11][0] = (vals[4]*s[4][0] + vals[7]*s[7][0])/(vals[4]+vals[7]); v[11][1] = (vals[4]*s[4][1] + vals[7]*s[7][1])/(vals[4]+vals[7]);
            
            v[0][2] = (vals[0]*s[0][2] + vals[1]*s[1][2])/(vals[0]+vals[1]);
            v[1][2] = (vals[1]*s[1][2] + vals[2]*s[2][2])/(vals[1]+vals[2]);
            v[2][2] = (vals[2]*s[2][2] + vals[3]*s[3][2])/(vals[2]+vals[3]);
            v[3][2] = (vals[0]*s[0][2] + vals[3]*s[3][2])/(vals[0]+vals[3]);
            v[4][2] = (vals[1]*s[1][2] + vals[5]*s[5][2])/(vals[1]+vals[5]);
            v[5][2] = (vals[2]*s[2][2] + vals[6]*s[6][2])/(vals[2]+vals[6]);
            v[6][2] = (vals[3]*s[3][2] + vals[7]*s[7][2])/(vals[3]+vals[7]);
            v[7][2] = (vals[0]*s[0][2] + vals[4]*s[4][2])/(vals[0]+vals[4]);
            v[8][2] = (vals[4]*s[4][2] + vals[5]*s[5][2])/(vals[4]+vals[5]);
            v[9][2] = (vals[5]*s[5][2] + vals[6]*s[6][2])/(vals[5]+vals[6]);
            v[10][2] = (vals[6]*s[6][2] + vals[7]*s[7][2])/(vals[6]+vals[7]);
            v[11][2] = (vals[4]*s[4][2] + vals[7]*s[7][2])/(vals[4]+vals[7]);
            
            // add new faces in the list
            for (int f = 0; f < nb_faces; f++) {
                    if (reverse) {
                        Faces[3*(offset+f)] = 3*(offset+f)+2;
                        Faces[3*(offset+f) +1] = 3*(offset+f)+1;
                        Faces[3*(offset+f) + 2] = 3*(offset+f);
                    } else {
                        Faces[3*(offset+f)] = 3*(offset+f);
                        Faces[3*(offset+f) +1] = 3*(offset+f)+1;
                        Faces[3*(offset+f) + 2] = 3*(offset+f)+2;
                    }
                    
                    Vertices[9*(offset+f)] = v[Config[index][f][0]][0];
                    Vertices[9*(offset+f)+1] = v[Config[index][f][0]][1];
                    Vertices[9*(offset+f)+2] = v[Config[index][f][0]][2];
                    
                    Vertices[9*(offset+f)+3] = v[Config[index][f][1]][0];
                    Vertices[9*(offset+f)+4] = v[Config[index][f][1]][1];
                    Vertices[9*(offset+f)+5] = v[Config[index][f][1]][2];
                    
                    Vertices[9*(offset+f)+6] = v[Config[index][f][2]][0];
                    Vertices[9*(offset+f)+7] = v[Config[index][f][2]][1];
                    Vertices[9*(offset+f)+8] = v[Config[index][f][2]][2];
                    
                    /*Normals[9*(offset+f)] = v[Config[index][f][0]][0];
                    Normals[9*(offset+f)+1] = v[Config[index][f][0]][1];
                    Normals[9*(offset+f)+2] = v[Config[index][f][0]][2];
                    
                    Normals[9*(offset+f)+3] = v[Config[index][f][1]][0];
                    Normals[9*(offset+f)+4] = v[Config[index][f][1]][1];
                    Normals[9*(offset+f)+5] = v[Config[index][f][1]][2];
                    
                    Normals[9*(offset+f)+6] = v[Config[index][f][2]][0];
                    Normals[9*(offset+f)+7] = v[Config[index][f][2]][1];
                    Normals[9*(offset+f)+8] = v[Config[index][f][2]][2];*/
            }
        }
    }
}
"""

Preambule_MCIndexing = """
//    create the precalculated 256 possible polygon configuration

__constant int ConfigCount[128] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 
2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 
3, 2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 
3, 2, 3, 2, 2, 1 };
"""

Kernel_MarchingCubeIndexing = """
int x = i/(dim_z*dim_y); 
int y = (i%(dim_z*dim_y))/dim_z; 
int z = (i%(dim_z*dim_y)) % dim_z;

if (x < dim_x-1 && y < dim_y-1 && z < dim_z-1) {

    // get the 8  current summits
    int s[8][3] = {{x, y, z}, {x+1,y , z}, {x+1, y+1, z}, {x, y + 1, z},
         {x, y, z+1}, {x+1, y, z+1}, {x+1, y+1, z+1}, {x, y+1, z+1}};
    float vals[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    
    // get the values of the implicit function at the summits
    // [val_0 ... val_7]
    bool stop = false;
    for (int k=0; k < 8; k++) {
        vals[k] = convert_float(TSDF[s[k][2] + dim_z*s[k][1] + dim_z*dim_y*s[k][0]])/30000.0f;
        if (fabs(vals[k]) >= 1.0f) {
            IndexVal[i] = 0;
            stop = true;
            break;
        }
    }
        
    if (!stop) {
        int index = 0;
        // get the index value corresponding to the implicit function
        if (vals[7] <= iso) {
            index = (int)(vals[0] > iso) + 
                (int)(vals[1] > iso)*2 + 
                (int)(vals[2] > iso)*4 + 
                (int)(vals[3] > iso)*8 + 
                (int)(vals[4] > iso)*16 + 
                (int)(vals[5] > iso)*32 + 
                (int)(vals[6] > iso)*64;
            IndexVal[i] = convert_short(index);
        } else{  
            index = (int)(vals[0] <= iso) + 
                (int)(vals[1] <= iso)*2 + 
                (int)(vals[2] <= iso)*4 + 
                (int)(vals[3] <= iso)*8 + 
                (int)(vals[4] <= iso)*16 + 
                (int)(vals[5] <= iso)*32 + 
                (int)(vals[6] <= iso)*64;
            IndexVal[i] = convert_short(-index);
        }
            
        // get the corresponding configuration
        if (index != 0) 
            Offset[i] = atomic_add(&faces_counter[0], ConfigCount[index]);
    }
}
"""

Kernel_OrganizeFaces = """

int summit = Faces[i];

float3 vtx = (float3) {Vertices[3*summit],  Vertices[3*summit+1], Vertices[3*summit+2]};

float diff;
int idx = nb_vertices/2;
float3 vtx_curr = (float3){0.0f,0.0f,0.0f};

int a = 0;
int b = nb_vertices;
while (b>a) {
    idx = (b+a)/2;
    vtx_curr.x = Unique_Vertices[3*idx];
    vtx_curr.y = Unique_Vertices[3*idx+1];
    vtx_curr.z = Unique_Vertices[3*idx+2];
    diff = fabs(vtx_curr.x - vtx.x) + fabs(vtx_curr.y - vtx.y) + fabs(vtx_curr.z - vtx.z);
    if (diff == 0.0f) 
        break;
        
    if (vtx.x == vtx_curr.x) {
        a = idx;
        while (vtx.x == vtx_curr.x && a > -1) {
            vtx_curr.x = Unique_Vertices[3*a];
            a--;
        }
        b = idx;
        while (vtx.x == vtx_curr.x && b < nb_vertices) {
            vtx_curr.x = Unique_Vertices[3*b];
            b++;
        }
        
        for (int l = a+1; l < b; l++) {
            idx = l;
            vtx_curr.x = Unique_Vertices[3*l];
            vtx_curr.y = Unique_Vertices[3*l+1];
            vtx_curr.z = Unique_Vertices[3*l+2];
            diff = fabs(vtx_curr.x - vtx.x) + fabs(vtx_curr.y - vtx.y) + fabs(vtx_curr.z - vtx.z);
            if (diff == 0.0f)
                break;
        }
        break;
    }
        
    if (vtx.x < vtx_curr.x) {
        b = idx;
    } else {
        a = idx;
    }
}

if (diff > 0.0f) {
    for (int k = 0; k < nb_vertices; k++) { 
        idx = k;
        vtx_curr.x = Unique_Vertices[3*idx];
        vtx_curr.y = Unique_Vertices[3*idx+1];
        vtx_curr.z = Unique_Vertices[3*idx+2];
        diff = fabs(vtx_curr.x - vtx.x) + fabs(vtx_curr.y - vtx.y) + fabs(vtx_curr.z - vtx.z);
        if (diff == 0.0f)
            break;
    }
}

Faces[i] = idx; 

"""

Kernel_ComputeNormales = """
if (i%3 == 0) {
    float3 pt[3];
    for (int k = 0; k < 3; k++) {
        pt[k].x = Vertices[3*Faces[i+k]];
        pt[k].y = Vertices[3*Faces[i+k]+1];
        pt[k].z = Vertices[3*Faces[i+k]+2];
    }
    
    float3 v1 = (float3)(pt[1].x-pt[0].x, pt[1].y-pt[0].y, pt[1].z-pt[0].z);
    float3 v2 = (float3)(pt[2].x-pt[0].x, pt[2].y-pt[0].y, pt[2].z-pt[0].z);
    float3 nmle = (float3)(v1.y*v2.z - v1.z*v2.y,
                         -v1.x*v2.z + v1.z*v2.x,
                         v1.x*v2.y - v1.y*v2.x);
    for (int k = 0; k < 3; k++) {
        atomic_add(&Normales[3*Faces[i+k]], convert_int(round(nmle.x*1000000000000.0f)));
        atomic_add(&Normales[3*Faces[i+k]+1], convert_int(round(nmle.y*1000000000000.0f)));
        atomic_add(&Normales[3*Faces[i+k]+2], convert_int(round(nmle.z*1000000000000.0f)));
    }
}
"""

Kernel_Normalise = """
if (i%3 == 0) {
    float norm = sqrt(Normales[i]*Normales[i] + Normales[i+1]*Normales[i+1] + Normales[i+2]*Normales[i+2]);
    if (norm > 0.0f) {
        Normales[i] = Normales[i]/norm;
        Normales[i+1] = Normales[i+1]/norm;
        Normales[i+2] = Normales[i+2]/norm;
    }
}
"""