from src.logger import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import sys
from numpy import genfromtxt,savetxt




@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self) -> None:
        self.scalerItem = StandardScaler()
        self.scalerUser = StandardScaler()
        self.scalerTarget = MinMaxScaler((-1,1))

    def intiate_transform(self,Itrain,Itest,Utrain,Utest,y_train,y_test):
        try:
            logging.info("Scaling the train and test data")
            unscaled_I_train = genfromtxt(Itrain,delimiter=',')
            unscaled_I_test = genfromtxt(Itest,delimiter=',')
            unscaled_U_train = genfromtxt(Utrain,delimiter=',')
            unscaled_U_test = genfromtxt(Utest,delimiter=',')
            unscaled_y_train = genfromtxt(y_train)
            unscaled_y_test = genfromtxt(y_test)

            self.scalerItem.fit(unscaled_I_train)
            scaled_item_train = self.scalerItem.transform(unscaled_I_train)
            scaled_item_test = self.scalerItem.transform(unscaled_I_test)
            self.scalerUser.fit(unscaled_U_train)
            scaled_user_train = self.scalerUser.transform(unscaled_U_train)
            scaled_user_test = self.scalerUser.transform(unscaled_U_test)
            self.scalerTarget.fit(unscaled_y_train)
            scaled_y_train = self.scalerTarget.transform(unscaled_y_train.reshape(-1,1))
            scaled_y_test = self.scalerTarget.transform(unscaled_y_test.reshape(-1,1))

            return(
                scaled_item_train,scaled_item_test,scaled_user_train,scaled_user_test,scaled_y_train,scaled_y_test
            )

        except Exception as e:
            raise CustomException(e,sys)



