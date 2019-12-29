{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np #Numeric\n",
    "import pandas as pd #Reading_file\n",
    "import sklearn #Ml Models\n",
    "import seaborn as sns #Visualization\n",
    "from sklearn.preprocessing import StandardScaler \n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt #Visualization\n",
    "%matplotlib inline\n",
    "from sklearn.linear_model import ElasticNet\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>mpg</th>\n",
       "      <th>cyl</th>\n",
       "      <th>disp</th>\n",
       "      <th>hp</th>\n",
       "      <th>drat</th>\n",
       "      <th>wt</th>\n",
       "      <th>qsec</th>\n",
       "      <th>vs</th>\n",
       "      <th>am</th>\n",
       "      <th>gear</th>\n",
       "      <th>carb</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Mazda RX4</td>\n",
       "      <td>21.0</td>\n",
       "      <td>6</td>\n",
       "      <td>160.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.90</td>\n",
       "      <td>2.620</td>\n",
       "      <td>16.46</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>Mazda RX4 Wag</td>\n",
       "      <td>21.0</td>\n",
       "      <td>6</td>\n",
       "      <td>160.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.90</td>\n",
       "      <td>2.875</td>\n",
       "      <td>17.02</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>Datsun 710</td>\n",
       "      <td>22.8</td>\n",
       "      <td>4</td>\n",
       "      <td>108.0</td>\n",
       "      <td>93</td>\n",
       "      <td>3.85</td>\n",
       "      <td>2.320</td>\n",
       "      <td>18.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>Hornet 4 Drive</td>\n",
       "      <td>21.4</td>\n",
       "      <td>6</td>\n",
       "      <td>258.0</td>\n",
       "      <td>110</td>\n",
       "      <td>3.08</td>\n",
       "      <td>3.215</td>\n",
       "      <td>19.44</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>Hornet Sportabout</td>\n",
       "      <td>18.7</td>\n",
       "      <td>8</td>\n",
       "      <td>360.0</td>\n",
       "      <td>175</td>\n",
       "      <td>3.15</td>\n",
       "      <td>3.440</td>\n",
       "      <td>17.02</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  \\\n",
       "0          Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4   \n",
       "1      Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4   \n",
       "2         Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4   \n",
       "3     Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3   \n",
       "4  Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3   \n",
       "\n",
       "   carb  \n",
       "0     4  \n",
       "1     4  \n",
       "2     1  \n",
       "3     1  \n",
       "4     2  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Mtcars = pd.read_csv(\"D:\\\\DATA SCIENCE\\\\Data Sets\\\\EXCEL FILES\\\\mtcars.csv\")\n",
    "Mtcars.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Mtcars.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mtcars = Mtcars.drop('Unnamed: 0', axis = 1)\n",
    "mtcars.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = mtcars.iloc[:,1:10]\n",
    "y = mtcars.iloc[:,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lm_ElasticNet = ElasticNet()\n",
    "\n",
    "lm_ElasticNet.fit(train_x, train_y)\n",
    "print(lm_ElasticNet.intercept_)\n",
    "print(lm_ElasticNet.coef_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.api as sm\n",
    "x2=sm.add_constant(x)\n",
    "sm.OLS(y,x2).fit().summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_score_lm_ElasticNet=lm_ElasticNet.score(train_x,train_y)\n",
    "test_score_lm_ElasticNet=lm_ElasticNet.score(test_x,test_y)\n",
    "coeff_used_lm_ElasticNet = np.sum(lm_ElasticNet.coef_!=0)\n",
    "print (\"ElasticNet_training score:\", train_score_lm_ElasticNet) \n",
    "print (\"ElasticNet_test score: \", test_score_lm_ElasticNet)\n",
    "print (\"ElasticNet_number of features used: \", coeff_used_lm_ElasticNet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (\"RSquare Value for ElasticNet Regression TEST data is-\")\n",
    "np.round(lm_ElasticNet.score(test_x,test_y)*100,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predict_test_ElasticNet = lm_ElasticNet.predict(test_x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (\"ElasticNet Regression Mean Square Error (MSE) for TEST data is\")\n",
    "np.round(metrics.mean_squared_error(test_y, predict_test_ElasticNet),2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MTCARS DATA MODEL\n",
    "x = drat,qsec,vs,am,gear\n",
    "\n",
    "y = mpg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = mtcars.iloc[:,[4,6,7,8,9]]\n",
    "Y = mtcars.iloc[:,0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_X, test_X, train_Y, test_Y = train_test_split(X,Y,test_size=0.3, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "LM_ElasticNet = ElasticNet()\n",
    "LM_ElasticNet.fit(train_X, train_Y)\n",
    "print(LM_ElasticNet.intercept_)\n",
    "print(LM_ElasticNet.coef_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import statsmodels.api as sm\n",
    "X1=sm.add_constant(X)\n",
    "sm.OLS(Y,X1).fit().summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_score_LM_ElasticNet=LM_ElasticNet.score(train_X,train_Y)\n",
    "test_score_LM_ElasticNet=LM_ElasticNet.score(test_X,test_Y)\n",
    "coeff_used_LM_ElasticNet = np.sum(LM_ElasticNet.coef_!=0)\n",
    "print (\"ElasticNet_training score:\", train_score_LM_ElasticNet) \n",
    "print (\"ElasticNet_test score: \", test_score_LM_ElasticNet)\n",
    "print (\"ElasticNet_number of features used: \", coeff_used_LM_ElasticNet)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (\"RSquare Value for ElasticNet Regression TEST data is-\")\n",
    "np.round(LM_ElasticNet.score(test_X,test_Y)*100,2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predict_test_LM_ElasticNet = LM_ElasticNet.predict(test_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (\"ElasticNet Regression Mean Square Error (MSE) for TEST data is\")\n",
    "np.round(metrics.mean_squared_error(test_Y, predict_test_LM_ElasticNet),2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
