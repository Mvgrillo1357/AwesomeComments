'''
import tensorflow as tf

class Model:
    def __init__(self, filepath, resize, save_type):
        #self.fold_name = filepath
        #self.resize = resize
        self.model_dir = "./models/" + self.fold_name
        # Loading the saved model from directory
        if (save_type == 'pickle'):
            self.model = tf.saved_model.load(self.model_dir)
        elif (save_type == 'h5'):
            self.model = tf.keras.models.load_model(self.model_dir)
        

    # Preprocess and run text through model, returns a map to results
    def process_txt(self, txt):
            
            # ADJUST SIZE PARAMETERS BASED ON MODEL
            
            # Normalize data
            
            
            # Putting image into model
            result = self.model(txt)
        
            # Convert result to a numpy array
            return result[0].numpy()
    
# Converts numpy_array results into mapped data with labels
def map_result(np_arr):

    # 6 values which represent what fruits our model classifies
    cyberbullying = ['religion', 'age', 'gender', 'ethnicity', 'not_cyberbullying', 'other_cyberbullying']
    sentiment = ['positive', 'negative']
    
    mapping = {}

    for (key, value) in zip(cyberbullying, np_arr):
        mapping[key] = value.item()

    return mapping
'''