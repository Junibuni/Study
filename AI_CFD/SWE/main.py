"""
pseudocode

encoder = Encoder()
decoder = Decoder()

time_integrator = LinearNet()

variables:
    sim_time: simulation time in sec
    manhole_data: manhole data with time

dt = 10s

ct = encoder(init)

for t in range((sim_time+10)//dt):
    xt = torch.cat([ct, ref_value], dim=1)

    dz = time_integrator(xt)

    z_t1 = dz + ct[:, :26]
    c_t1 = torch.cat((z_t1, ref_value), dim=1)
    
    sim_out = decoder(c_t1)
    save sim_out
    
    ct = c_t1
"""