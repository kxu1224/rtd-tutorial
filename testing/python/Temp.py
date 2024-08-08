def PilotExperiment(dataname, pilot_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None):
    """ 
        This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
        with several pilot size given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
        for each pilot size, there will be 5 draws, 
        for each draw, the data is augmented to 5 times the original sample size.
        dataname :         pure data name without .csv. Eg: SKCMPositive_3
        pilot_size:        a set including potential pilot sizes
        model:             name of the model to be trained
        batch_frac:        batch fraction
        learning_rate:     learning rate 
        epoch:             choose from None (early_stop), or any integer, if choose None, early_stop_num will take effect
        early_stop_num:    if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
        off_aug:           choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
        AE_head_num:       how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
        Gaussian_head_num: how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
        pre_model:         transfer learning input model. If pre_model == None, no transfer learning 
    """
    print("Pilot Experiment Start")
    return None

#%% Define application of experiment
def ApplyExperiment(path, dataname, apply_log, new_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None, save_model = None):
# This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
#      given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
#      and generate new samples with size specified by the users.
# path:              path for reading real data and saving new data
# dataname :         pure data name without .csv. Eg: SKCMPositive_3
# apply_log:         logical whether apply log transformation before training
# model:             name of the model to be trained
# batch_frac:        batch fraction
# learning_rate:     learning rate 
# epoch:             choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
# early_stop_num:    if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
# off_aug:           choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
# AE_head_num:       how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
# Gaussian_head_num: how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
# pre_model:         transfer learning input model. If pre_model == None, no transfer learning
# save_model:        if the trained model should be saved, specify the path and name of the saved model
    print("Apply Experiment Start")
    return None

#%% Define transfer learing 
def Transfer(pilot_size, fromname, toname, fromsize, model, new_size=500, apply_log=True, epoch=None, batch_frac=0.1, learning_rate=0.0005, off_aug=None):
# This function run transfer learning using VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
#      given fromdata, todata, model, batch_size, learning_rate, epoch, off_aug and pre_model
#      and generate new samples with size specified by the users.
#      The fine tuning model training can be pilot experiments or apply experiment
# Make sure data files for pre_model training and fine tuning model training are in Transfer/
# pilot_size:        if None, the fine tuning model will be apply experiment and new_size will take effect
#                    otherwise, the fine tuning model will be trained using pilot experiments
# fromname:          the dataname for pre_model training 
# toname:            the dataname for fine tuning model training
# fromsize:          the sample size of the fromdata
# new_size:          if apply experiment, this will be the sample size of generated samples
# apply_log:         logical whether apply log transformation before training
# model:             name of the model to be trained
# batch_frac:        batch fraction
# learning_rate:     learning rate 
# epoch:             choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
# off_aug:           choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation

    print("Transfer Learning Start")
    return None