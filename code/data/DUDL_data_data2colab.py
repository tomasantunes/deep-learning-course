import torchvision

# download the CIFAR10 dataset
cdata = torchvision.datasets.CIFAR10(root='cifar10', download=True)

print(cdata)

# Datasets that come with torchvision: https://pytorch.org/vision/stable/index.html

import pandas as pd

# url
marriage_url = 'https://www.cdc.gov/nchs/data/dvs/state-marriage-rates-90-95-99-19.xlsx'

# import directly into pandas
data = pd.read_excel(marriage_url,header=5)
data



from google.colab import files
uploaded = files.upload()



from google.colab import drive
drive.mount('/content/gdrive')


