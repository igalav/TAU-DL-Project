import imageio
import numpy as np
import os
from StyleBank import StyleBank

SB = StyleBank()    ## Create Instance of the StyleBank class
#SB.Use_Batch_Norm = False  ## Choose notmalization layer (batch/instance)
SB.initialize_placeholders() ## Initialize Place holders
SB.build_models() ## Create the models (AutoEncoder + Style networks)
SB.compile_models() ## Compile the models
SB.prepare_tensorboard() ## Tensorboard
SB.train_models() ## Train the models
SB.save_models() ## Save the models
SB.load_models() ## Loads models from Save_BatchNorm/Save_InstNorm directories
print ("Final LR: {}".format(SB.LR_Current))
print ("Used Styles:")
print (SB.Style_DB_list)

out_img_dir = 'OutImages'
os.makedirs(out_img_dir)
AE_out = SB.AutoEncoderNet.predict(SB.Content_DB) ## Run the autoencoder
for i in range(SB.Batch_Size):
    imageio.imwrite(os.path.join(out_img_dir,"in_{}.png".format(i)), np.uint8(255.0*(SB.Content_DB[i] / np.max(SB.Content_DB[i]))))
    for k in range(SB.n_styles):
        SN_out = SB.StyleNet[k].predict(SB.Content_DB) ## Run the style networks
        imageio.imwrite(os.path.join(out_img_dir,"SN{}_{}.png".format(k,i)), np.uint8(255.0 * (SN_out[i] / np.max(SN_out[i]))))
    imageio.imwrite(os.path.join(out_img_dir,"AEN_{}.png".format(i)), np.uint8(255.0 * (AE_out[i] / np.max(AE_out[i]))))
