import torch
from torchvision import transforms
import numpy as np
from dataloader import CustomDataset
from swe_ae import SWE_AE
from tqdm import tqdm

from utils import unnormalize
import matplotlib.pyplot as plt

ckpt_pth = r"D:\epoch=99-step=24300.ckpt"
checkpoint = torch.load(ckpt_pth)

model_input = dict(
    optim_params = dict(lr=2e-4),
    scheduler_params = dict(),
    input_size = (1, 3, 384, 256)
)
model = SWE_AE(**model_input)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to("cuda")

data = CustomDataset(r"AI_CFD\SWE\datasets", normp=True)

def extract_integer_from_last_path(x):
    return int(x['xvel'].split('\\')[-1].split('_')[-1].split('.')[0])
data.grouped_files = sorted(
    [d for d in data.grouped_files if '333' in d['xvel'].split('\\')[-2]],
    key=extract_integer_from_last_path
)

import vtk

def save_vtk_2d_array(filename, data):
    # Create vtkImageData
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape[1], data.shape[0], 1)
    vtk_data.SetSpacing(1.0, 1.0, 1.0)
    vtk_data.SetOrigin(0.0, 0.0, 0.0)
    vtk_data.AllocateScalars(vtk.VTK_DOUBLE, 1)

    # Fill vtkImageData with the 2D array
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            vtk_data.SetScalarComponentFromDouble(x, y, 0, 0, data[y, x])

    # Write vtkImageData to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_data)
    writer.Write()

anim = True
if anim:
    # with torch.no_grad():
    #     for i in tqdm(range(len(data))):
    #         img = data[i][0].to("cuda")
    #         # lvec = model.encoder(img.unsqueeze(0))
    #         # a = np.array(lvec.squeeze().cpu())
    #         # lvec_stack.append(a)
    #         # lvec_stack_rmv.append(a[:-6])
    #         result = model(img.unsqueeze(0))
    #         result = unnormalize(result[0]).cpu().numpy().squeeze()[23:, 47:-2]
            
    #         plt.imshow(result)
    #         plt.clim(0, 0.2)
    #         plt.axis(False)
    #         plt.imsave(f"video\{i}.png", result)
    #         quit()

    def get_frame(i):
        print(f"{i+1}/{len(data)}")
        with torch.no_grad():
            _img = data[i][0].to("cuda")
            _lv = data[i][1].to("cuda")
            lv = model.encoder(_img.unsqueeze(0))

            lv = torch.concat((lv, _lv[-6:].unsqueeze(0)), dim=1)
            
            result = model.decoder(lv)
            result = unnormalize(result[0]).cpu().numpy().squeeze()[23:, 47:-2]
            # result = unnormalize(_img[0]).numpy().squeeze()[23:, 47:-2]
            save_vtk_2d_array(fr"D:\swe\frame{i}.vti", result)
        return result


    fig, ax = plt.subplots(figsize=(4, 6))
    heatmap = ax.imshow(get_frame(0))
    ax.axis(False)
    heatmap.set_clim(0, 0.2)
    fig.colorbar(heatmap, ax=ax)
    frame_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")
    plt.show()

    def update(frame):
        heatmap.set_array(get_frame(frame))
        frame_text.set_text(f"time: {frame*10}s")
        return [heatmap, frame_text]
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(data), interval=100, blit=True)

    animation_filename = "333_pred.gif"
    ani.save(animation_filename, writer='pillow')


    plt.close(fig)
else:
    lvec_stack, lvec_stack_rmv = [], []
    for i in tqdm(range(len(data))):
        with torch.no_grad():
            _img = data[i][0].to("cuda")
            # _lv = data[i][1].to("cuda")
            lv = model.encoder(_img.unsqueeze(0))
            a = np.array(lv.squeeze().cpu())
            lvec_stack.append(a)
            lvec_stack_rmv.append(a[:-6])

    plt.figure()
    plt.imshow(np.array(lvec_stack).transpose())
    plt.title("132")
    plt.colorbar(orientation='horizontal')
    plt.show()

    plt.figure()
    plt.imshow(np.array(lvec_stack_rmv).transpose())
    plt.title("132 rmv")
    plt.colorbar(orientation='horizontal')
    plt.show()
quit()

import matplotlib.pyplot as plt
with torch.no_grad():
    print(img.shape)
    imgcrop = img.clone()[0:1, :, :].permute(1, 2, 0)[23:, 47:-2]
    print(imgcrop.shape)
    plt.imshow(unnormalize(imgcrop))
    plt.clim(0, 0.2)
    plt.show()

    out, lv = model(img.unsqueeze(0))
    print(lv, lv.shape)
    print(out, out.shape)
    out_numpy = unnormalize(out.clone().detach()).numpy().squeeze()
    plt.imshow(out_numpy[23:, 47:-2])
    plt.clim(0, 0.2)
    plt.show()



    rmse_error = np.sqrt((unnormalize(out).detach().numpy().squeeze() - unnormalize(img[0:1, :, :]).detach().numpy().squeeze())**2)
    print(rmse_error.shape)
    plt.imshow(rmse_error[23:, 47:-2])
    plt.colorbar()
    plt.show()

    rsmpe_error = np.sqrt((unnormalize(out).detach().numpy().squeeze() - unnormalize(img[0:1, :, :]).detach().numpy().squeeze())**2/unnormalize(img[0:1, :, :]).squeeze())*100
    plt.imshow(rsmpe_error[23:, 47:-2])
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()
