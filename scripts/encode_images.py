from encode_images import ImgEncoder

if __name__ == "__main__":
    # instantiate Image encoder using trained model
    images_encoder = ImgEncoder(
        encoder_name="<path>.h5", 
        load=True, 
        encoder_model=None)
    
    # Get all the data from folder
    x_data, data_fr = images_encoder.read_imgs_to_np_from_folder(0, 200000)
    
    # Insert encodings into dataframe
    product_codes_df = images_encoder.encode_all(x_data, data_fr)

    # Insert to database
    images_encoder.insert_to_db(product_codes_df)