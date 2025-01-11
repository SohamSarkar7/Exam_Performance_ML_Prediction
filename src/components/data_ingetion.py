import os
import sys
from src.exception import CustomeException
from src.loggers import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngetionconfig:

    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"data.csv")


class Dataingetion:
    def __init__(self):
        self.ingetion_config=DataIngetionconfig()

    def initiate_data_function(self):
        logging.info('Entered the data ingetion methodo or component')

        try:
            df = pd.read_csv('notebook/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingetion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingetion_config.raw_data_path,index=False,header=True)


            logging.info("Train test split initialized")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingetion_config.train_data_path,index=False,header = True)
            test_set.to_csv(self.ingetion_config.test_data_path,index=False,header = True)

            logging.info("Ingetion of the data is completed")

            return(
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path
            )

        except Exception as e:

            raise CustomeException(e,sys)
        

if __name__ == "__main__":
    obj = Dataingetion()
    obj.initiate_data_function()




    