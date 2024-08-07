def PilotExperiment(dataname, pilot_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None):
# This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
#      with several pilot size given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
#      for each pilot size, there will be 5 draws, 
#      for each draw, the data is augmented to 5 times the original sample size.
# dataname :         pure data name without .csv. Eg: SKCMPositive_3
# pilot_size:        a set including potential pilot sizes
# model:             name of the model to be trained
# batch_frac:        batch fraction
# learning_rate:     learning rate 
# epoch:             choose from None (early_stop), or any integer, if choose None, early_stop_num will take effect
# early_stop_num:    if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
# off_aug:           choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
# AE_head_num:       how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
# Gaussian_head_num: how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
# pre_model:         transfer learning input model. If pre_model == None, no transfer learning
    print("Pilot Experiment Start")
    return None