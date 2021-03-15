from sklearn.utils import resample
import pandas as pd
import re
import os

class AgeDataset():
  def __init__(self, directory):
    """
    Params:
    Directory -> Should be the path to the folder containing all 
    images of the dataset, named as the original set.

    Return:
    The return is given by the dataframe_gen function, which
    provides 

    """
    self.directory = directory
    self.minority_length = 2397
    self.image_pattern = re.compile(r'\b(\d{1,3})_')
    self.image_df = pd.DataFrame(columns = ['image_name','label'])

  def age_ranges(self):
    """
    Divide our ages in a few ranges:
    From 0 to 10 -> CRIANCA
    From 10 to 24 -> JOVEM
    From 24 to 60 -> ADULTO
    Everything above 60 -> IDOSO
    """
    range = [] 
    for age in self.image_df['label']:
      if age <= 10: 
        range.append('CRIANCA')

      elif (age > 10) & (age <= 24):
        range.append('JOVEM')

      elif (age > 24) & (age <= 60):
        range.append('ADULTO')

      else: 
        range.append('IDOSO')

    self.image_df['age_div'] = range


  def print_distribution(self, df):
    """
    Prints the distribution over the classes in our dataset;
    """
    print('\nClass distribution:\n')
    print('Idosos: {}'.format(len(df[df['age_div'] == 'IDOSO'])))
    print('Crianças: {}'.format(len(df[df['age_div'] == 'CRIANCA'])))
    print('Jovens: {}'.format(len(df[df['age_div'] == 'JOVEM'])))
    print('Adultos: {}\n'.format(len(df[df['age_div'] == 'ADULTO'])))


  def GenerateDFrame(self):
    print('Loading and treating data...')
    for image in os.listdir(self.directory):
      label = self.image_pattern.findall(image)[0]
      self.image_df = self.image_df.append({'image_name': image, 'label': int(label)}, ignore_index = True)

    self.age_ranges()
    self.print_distribution(self.image_df)
    
    print('Done!')
    return self.image_df.reset_index(drop=True)








