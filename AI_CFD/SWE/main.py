# """
# pseudocode

# encoder = Encoder()
# decoder = Decoder()

# time_integrator = LinearNet()

# variables:
#     sim_time: simulation time in sec
#     manhole_data: manhole data with time

# dt = 10s

# ct = encoder(init)

# for t in range((sim_time+10)//dt):
#     xt = torch.cat([ct, ref_value], dim=1)

#     dz = time_integrator(xt)

#     z_t1 = dz + ct[:, :26]
#     c_t1 = torch.cat((z_t1, ref_value), dim=1)
    
#     sim_out = decoder(c_t1)
#     save sim_out
    
#     ct = c_t1
# """
# import os 

# import torch
# import vtk
# import numpy as np
# from torchvision.transforms import transforms

# from swe_ae import SWE_AE, ManifoldNavigator
# from utils import unnormalize

# def save_vtk(filename, data):
#     data = data.squeeze().squeeze()
#     vtk_data = vtk.vtkImageData()
#     vtk_data.SetDimensions(data.shape[1], data.shape[0], 1)
#     vtk_data.SetSpacing(1.0, 1.0, 1.0)
#     vtk_data.SetOrigin(0.0, 0.0, 0.0)
#     vtk_data.AllocateScalars(vtk.VTK_DOUBLE, 1)

#     for y in range(data.shape[0]):
#         for x in range(data.shape[1]):
#             vtk_data.SetScalarComponentFromDouble(x, y, 0, 0, data[y, x])

#     writer = vtk.vtkXMLImageDataWriter()
#     writer.SetFileName(filename)
#     writer.SetInputData(vtk_data)
#     writer.Write()

# ckpt_pth_ae = r"D:\Study\AI_CFD\SWE\logs\epoch=917-step=223074.ckpt"
# checkpoint_ae = torch.load(ckpt_pth_ae)
# ae_model = SWE_AE(input_size = (1, 3, 384, 256), cnum=64)
# ae_model.load_state_dict(checkpoint_ae['state_dict'])

# ckpt_pth_linear = r"D:\Study\AI_CFD\SWE\logs\epoch=3870-step=1873564.ckpt"
# checkpoint_linear = torch.load(ckpt_pth_linear)
# linear_model = ManifoldNavigator(batch_size=32, hidden_shape = 28, model_type="original", cnum=64+6)
# linear_model.load_state_dict(checkpoint_linear['state_dict'])

# ae_model.eval()
# linear_model.eval()

# encoder = ae_model.encoder
# decoder = ae_model.decoder

# with torch.no_grad():
#     file_path = 'ref.csv'
#     ref_value = np.loadtxt(file_path, delimiter=',') 
#     # =================
#     save_pth = r"D:\swe"
#     sim_time = 3600
#     dt = 10
#     # =================
#     def norm_p(p, mean=0.2188, std=0.6068): 
#         normalized_p = (p - mean) / std
#         return normalized_p

#     norm_img = transforms.Normalize((0.0059, 0.0004, -0.0045),(0.0198, 0.0300, 0.0297))
#     init = torch.zeros(1, 3, 384, 256)
#     init = norm_img(init)
#     ct = encoder(init)
#     # ct = torch.zeros(1, 26)

#     xt = torch.cat([ct, norm_p(torch.tensor(ref_value[0]).unsqueeze(0))], dim=1).float()
#     sim_out = decoder(xt)
#     save_vtk(os.path.join(save_pth, "frame0.vti"), sim_out)



#     lvec_stack = []
#     for t in range(1, sim_time//dt):
#         print(f"Time: {t*10}sec", end='\r')
#         #xt = torch.cat([ct, norm_p(torch.tensor(ref_value[t-1]).unsqueeze(0))], dim=1).float()
#         xt = torch.cat([ct, norm_p(torch.tensor(ref_value[t-1]).unsqueeze(0)), norm_p(torch.tensor(ref_value[t]).unsqueeze(0))-norm_p(torch.tensor(ref_value[t-1]).unsqueeze(0))], dim=1).float()

#         dz = linear_model(xt.unsqueeze(0))

#         #c_t1 = torch.cat([dz, norm_p(torch.tensor(ref_value[t]).unsqueeze(0))], dim=1).float()
#         c_t1 = torch.cat([ct+dz, norm_p(torch.tensor(ref_value[t]).unsqueeze(0))], dim=1).float()
        
#         sim_out = decoder(c_t1)
        
#         result = unnormalize(sim_out)
#         save_vtk(os.path.join(save_pth, f"frame{t}.vti"), result)
        
#         ct = dz

#         a = np.array(xt.squeeze().cpu())
#         lvec_stack.append(a)
# np.save("good.npy", lvec_stack)
###############################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# legend_elements = [Line2D([0], [0], lw=1, color='r', label='231'),
#                    Line2D([0], [0], lw=1, color='g', label='233'),
#                    Line2D([0], [0], lw=1, color='b', label='533'),]

lvec_stack = np.load("good.npy")

i= 1
plt.figure(figsize=(20, 50))
for group in range(64):
    plt.subplot(64, 1, i)
    plt.plot(lvec_stack[:, group], color="r")
    i += 1

# plt.legend(handles=legend_elements, loc='best')
plt.show()
###############################        