import numpy as np
import matplotlib.pyplot as plt
from env.VF_bases import uni_xflow, uni_yflow, fixed_vortex, free_vortex, sink, source, doublet
from env.VF_bases import rankine, corner_flow, saddle, field_noise, boundary_layer

#5. Define the Vector Field by summing different bases together.


def position_readings(x,y): # Fill out the Vector Field Below
    
    u = []
    v = []

   # Add Source/Sink
    #x_source1 = 2
    #y_source1 = 5
    #u1, v1 = source(x, y, x_source1, y_source1)
    #u.append(2*u1) # use +ve for Source/Spring ,  -ve for sink
    #v.append(2*v1) # use +ve for Source/Spring ,  -ve for sink
    
    #Add free Vortex
    # x_source2 = 2
    # y_source2 = 3.1 
    # u2, v2 = free_vortex(x, y, x_source2, y_source2)
    # #u2, v2 = fixed_vortex(x, y, x_source2, y_source2) 
    # u.append(u2)
    # v.append(v2)
    #Use -ve to get counterclockwise rotation. 
    
    # Add free Vortex
    # x_source2 = 7.5
    # y_source2 = 6.2 
    # u22, v22 = free_vortex(x, y, x_source2, y_source2)
    # #u2, v2 = fixed_vortex(x, y, x_source2, y_source2) 
    # u.append(-u22)
    # v.append(-v22)
    #Use -ve to get counterclockwise rotation. 
    
    # # Add Dipole
    # x_source3 = 5
    # y_source3 = 5 
    # u3, v3 = doublet(x, y, x_source3, y_source3) 
    # u.append(u3)
    # v.append(v3)
 
    #  Add rankine
    # x_source4 = 6
    # y_source4 = 6 
    # u4, v4 = rankine(x, y, x_source4, y_source4) 
    # #u4, v4 = free_vortex(x, y, x_source4, y_source4) 
    # u.append(u4)
    # v.append(v4)

    #  Add Boundary Layer
     
    u4, v4 = boundary_layer(x, y)  
    u.append(u4)
    v.append(v4)
    
    
# # # # Add Uniform flow x direction
#     print("x in vF_enf before unflow", x)

    u3, v3 = uni_xflow(x, y)
    #u.append(.2*u3)
    #v.append(.2*v3)    
    
# #     # Add Saddle
#     x_saddle = 5
#     y_saddle = 5
#     u4 , v4 = saddle(x, y, x_saddle, y_saddle)
#     u.append(u4)
#     v.append(v4)

    # Add some Sensor Noise
    # vary = 0.1
    # np.random.seed(None)
    # un = vary*np.random.randn()
    # vn = vary*np.random.randn()
    # u.append(un)
    # v.append(vn)    
    
    # ----------------------
    # # Adding Random Field
    # print("x in vF_enf", x)
    # u6, v6 = field_noise(x,y)
    # #print('U6 and V6', u6, v6)
    # if signal_noise ==0:
    #     signal_noise = 0.001
    # noise_factor = 1/signal_noise - 1
    # u.append(*np.array(u6))
    # v.append(0.1*np.array(v6))
    # print("u", u)
    # print("v",v)   

    # Add them altogether
    u = np.sum(np.array(u), axis=0) #+ np.random.uniform(-0.1,0.1)
    v = np.sum(np.array(v), axis=0) #+ np.random.uniform(-0.1,0.1)

    u = np.clip(u, 0.002, .15)
    v = np.clip(v, 0, 10)
    
    return u, v


