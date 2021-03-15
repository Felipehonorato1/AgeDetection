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
    self.image_pattern = re.compile(r'\b(\d{1,3})_')
    self.image_df = pd.DataFrame(columns = ['image_name','label'])

  def get_minority_length(self):
    return self.simple_df['age_div'].value_counts().min()

  def age_ranges(self):
    """
    Divide our ages in a few ranges:
    From 0 to 5-> BEBE
    From 5 to 14 -> CRIANÇA
    From 14 to 24 -> JOVEM
    From 24 to 60 -> ADULTO
    Everything above 60 -> IDOSO
    """
    range = [] 
    for age in self.image_df['label']:
      if age <= 5: 
        range.append('BEBE')

      elif (age > 5) & (age <= 14):
        range.append('CRIANCA')

      elif (age > 14) & (age <= 24):
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
    print('Bebês: {}'.format(len(df[df['age_div'] == 'BEBE'])))
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

  
  def dataSample(self,n ,seed = None):
    """
    Params: 
    seed = Random state that will be applied to our sample function
    n = Number of elements of each group 

    Returns:
    result_df with n*number_of_classes
    """


    bebe_sample = self.image_df[self.image_df['age_div'] == 'BEBE'].sample(n, random_state = seed)
    crianca_sample = self.image_df[self.image_df['age_div'] == 'CRIANCA'].sample(n, random_state = seed)
    jovem_sample = self.image_df[self.image_df['age_div'] == 'JOVEM'].sample(n, random_state = seed)
    adulto_sample = self.image_df[self.image_df['age_div'] == 'ADULTO'].sample(n, random_state = seed)
    idoso_sample = self.image_df[self.image_df['age_div'] == 'IDOSO'].sample(n, random_state = seed)
    result_df = pd.concat([bebe_sample, crianca_sample, jovem_sample, adulto_sample, idoso_sample], ignore_index= True, sort = True)
    self.print_distribution(result_df)

    return result_df


  def undersample(self, seed = None):
    """
    Params:
    Seed: Set the random_state that is going to be passed to the datasample function.
    
    Returns:
    A dataframe undersampled over the minority class count.

    """
    n = get_minority_length()
    undersampled_df = dataSample(n, seed = seed)
    return undersampled_df
    