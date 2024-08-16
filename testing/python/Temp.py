def PilotExperiment(dataname, pilot_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None):
    r""" 
        This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
        with several pilot size given data, model, batch_size, learning_rate, epoch, off_aug and pre_model.
        For each pilot size, there will be 5 draws.
        For each draw, the data is augmented to 5 times the original sample size.

        Parameters
        ----------
        dataname : string
                   pure data name without .csv. Eg: SKCMPositive_3
        pilot_size : list
                     a list including potential pilot sizes
        model : string
                name of the model to be trained
        batch_frac : float
                      batch fraction
        learning_rate : float
                learning rate 
        epoch : int
                                choose from None (early_stop), or any integer, if choose None, early_stop_num will take effect
        early_stop_num : int
              if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
        off_aug : string (AE_head or Gaussian_head or None)
                            choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
        AE_head_num : int
                    how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
        Gaussian_head_num : int
                how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
        pre_model : string
                        transfer learning input model. If pre_model == None, no transfer learning 

    """
    print("Pilot Experiment Start")
    return None

#%% Define application of experiment
def ApplyExperiment(path, dataname, apply_log, new_size, model, batch_frac, learning_rate, epoch, early_stop_num = 30, off_aug = None, AE_head_num = 2, Gaussian_head_num = 9, pre_model = None, save_model = None):
    r"""
        This function trains VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
        given data, model, batch_size, learning_rate, epoch, off_aug and pre_model
        and generate new samples with size specified by the users.

    Parameters
    ----------
    path : string
                              path for reading real data and saving new data
    dataname : string
                    pure data name without .csv. Eg: SKCMPositive_3
    apply_log : boolean
                      logical whether apply log transformation before training
    model : string
                              name of the model to be trained
    batch_frac : float
                    batch fraction
    learning_rate : float
              learning rate 
    epoch : int
                              choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
    early_stop_num : int
            if loss does not improve for early_stop_num epochs, the training will stop. Default value is 30. Only take effect when epoch == "early_stop"
    off_aug : string (AE_head or Gaussian_head or None) 
                      choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation
    AE_head_num : int
                  how many folds of AEhead augmentation needed. Default value is 2, Only take effect when off_aug == "AE_head"
    Gaussian_head_num : int
         how many folds of Gaussianhead augmentation needed. Default value is 9, Only take effect when off_aug == "Gaussian_head"
    pre_model : string
                      transfer learning input model. If pre_model == None, no transfer learning
    save_model : string
                    if the trained model should be saved, specify the path and name of the saved model
    """
    
    print("Apply Experiment Start")
    return None

#%% Define transfer learing 
def TransferExperiment(pilot_size, fromname, toname, fromsize, model, new_size=500, apply_log=True, epoch=None, batch_frac=0.1, learning_rate=0.0005, off_aug=None):
    """
        This function run transfer learning using VAE or CVAE, or GAN, WGAN, WGANGP, MAF, GLOW, RealNVP 
        given fromdata, todata, model, batch_size, learning_rate, epoch, off_aug and pre_model
        and generate new samples with size specified by the users.
        The fine tuning model training can be pilot experiments or apply experiment
        Make sure data files for pre_model training and fine tuning model training are in Transfer/

    Parameters
    ----------
    pilot_size : int
                    if None, the fine tuning model will be apply experiment and new_size will take effect
                    otherwise, the fine tuning model will be trained using pilot experiments
    fromname : string
                        the dataname for pre_model training 
    toname : string 
                            the dataname for fine tuning model training
    fromsize : int
                        the sample size of the fromdata
    new_size : int
                        if apply experiment, this will be the sample size of generated samples
    apply_log : boolean
                      logical whether apply log transformation before training
    model : string
                              name of the model to be trained
    batch_frac : float
                    batch fraction
    learning_rate : float
              learning rate 
    epoch : int
                              choose from None (early_stop), or any interger, if choose None, early_stop_num will take effect
    off_aug : string (AE_head or Gaussian_head or None)
                          choose from AE_head, Gaussian_head, None. if choose AE_head, AE_head_num will take effect. If choose Gaussian_head, Gaussian_head_num will take effect. If choose None, no offline augmentation"""

    print("Transfer Learning Start")
    return None

def heatmap_eval(dat_generated, dat_real):
    r"""
    This function creates a heatmap visualization comparing the generated data and the real data

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the data generated from ApplyExperiment
    dat_real: pd.DataFrame
            the original copy of the data
    
    """
    print("heatmap")
    return None

def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, legend_pos="top"):
    r"""
    This function creates a UMAP visualization comparing the generated data and the real data

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the data generated from ApplyExperiment
    dat_real: pd.DataFrame
            the original copy of the data
    groups_generated : pd.Series
            the groups generated
    groups_real : pd.Series
            the real groups
    legend_pos : string
            legend location
    
    """
    print("umap")
    return None

def evaluation(generated_input: str = "BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv", 
               real_input: str = "BRCASubtypeSel_test.csv"):
    r"""
    This method provides preprocessing of the input data prior to creating the visualizations.
    This can also be used as inspiration for other ways of using the above evaluation methods.

    generated_input : string
        the generated dataset; a default set is also provided as an example
    real_input : string
        the real original dataset; a default set is also provided as an example
    
    """
    print("evals!")
    return None