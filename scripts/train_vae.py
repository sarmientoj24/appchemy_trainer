from vae import VAE
from utils import read_imgs_to_np_from_folder


if __name__ == "__main__":
    # Get dataset
    X_train = read_imgs_to_np_from_folder('<path>')

    # Instantiate VAE model
    vae = VAE()

    # Train model
    vae.fit()

    # Plot loss function
    # vae.plot_loss_function()

    # Save trained model to be used for the webapp
    vae.get_vae_model().save('<path>.h5')
    vae.get_encoder_model().save('<path>.h5')
    vae.get_decoder_model().save('<path>.h5')