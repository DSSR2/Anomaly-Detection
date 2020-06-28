import os
from glob import glob
from Anomaly_Detector import AD

'''
Create and Train Model from Scratch
'''
ad = AD("AD1")
ad.load_data("Dataset/Train/")
ad.train(75)

'''
Load Trained Model and Train again on Different Data
'''
ad = AD("Models/model")
ad.load_data("Dataset2/Train")
ad.train(75)

'''
Check if Model Works Well
'''
ad.test_img("Dataset/Train/0.jpg")

'''
Load Model and Predict
'''
ad = AD("Models/model")
# Predict on file
ad.predict("Dataset/Test/0.jpg", "Output/", hardcore=True)
# Predict on fodler
ad.predict("Dataset/Test/", "Output/")