import pickle
import matplotlib.pyplot as plt

with open("DCGAN\checkpoint\losses\celeba.pkl", 'rb') as f: 
    d = pickle.load(f)

#["loss_G_train", "loss_D_real_train", "loss_D_fake_train", "loss_D_train"]
plt.figure()
#plt.plot(d['loss_G_train'], label="loss_G_train")
plt.plot(d['loss_D_real_train'], label="loss_D_real_train")
plt.plot(d['loss_D_fake_train'], label="loss_D_fake_train")
#plt.plot(d['loss_D_train'], label="loss_D_train")
plt.legend()
plt.show()