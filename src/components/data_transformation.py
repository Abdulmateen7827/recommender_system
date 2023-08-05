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
    def intiate_transform(self,Itrain,Itest,Utrain,Utest,y_train,y_test):
        try:
            logging.info("Scaling the train and test data")
            unscaled_I_train = genfromtxt(Itrain,delimiter=',')
            unscaled_I_test = genfromtxt(Itest,delimiter=',')
            unscaled_U_train = genfromtxt(Utrain,delimiter=',')
            unscaled_U_test = genfromtxt(Utest,delimiter=',')
            unscaled_y_train = genfromtxt(y_train)
            unscaled_y_test = genfromtxt(y_test)

            scalerItem = StandardScaler()
            scalerUser = StandardScaler()
            scalerTarget = MinMaxScaler((-1,1))

            scaled_item_train = scalerItem.fit_transform(unscaled_I_train)
            scaled_item_test = scalerItem.transform(unscaled_I_test)
            scaled_user_train = scalerUser.fit_transform(unscaled_U_train)
            scaled_user_test = scalerUser.transform(unscaled_U_test)

            scaled_y_train = scalerTarget.fit_transform(unscaled_y_train.reshape(-1,1))
            scaled_y_test = scalerTarget.transform(unscaled_y_test.reshape(-1,1))

            return(
                scaled_item_train,scaled_item_test,scaled_user_train,scaled_user_test,scaled_y_train,scaled_y_test
            )

        except Exception as e:
            raise CustomException(e,sys)



