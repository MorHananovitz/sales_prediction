# Sales Prediction

The goal it to create a sales prediction model for 5 different products.
Raw data contains:
* A company sales per product for the years Jan-2020 to May-2023
* Promotion data for 2020-2023
* Hoiliday calander for 2020-2023
* Competitor sales data - agregated for all products in Jan-2020 to May-2023

## Selected model
Due to the seasonality factor found in the products using `{seasonal_decompose}` and the two files that provided exhougenouse data - the selected model was chosen to be SARIMAX.

__SARIMAX Model__
SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) is an extension of the SARIMA model that incorporates additional exogenous variables to improve the forecasting performance. SARIMAX models are widely used for time series analysis and forecasting when the target variable is influenced by external factors.

All the needed SARIMAX components exist in the given raw data.

1. Seasonal Component (S): The seasonal component captures the repeating patterns or cycles in the data that occur over fixed intervals, such as daily, weekly, or monthly. It includes parameters for the seasonal order (P, D, Q) that control the seasonal autoregressive, differencing, and moving average components, respectively.

2. Trend Component (T): The trend component represents the long-term behavior or systematic change in the data over time. It includes parameters for the non-seasonal order (p, d, q) that control the autoregressive, differencing, and moving average components, respectively.

3. Error Component (E): The error component accounts for the random or unpredictable fluctuations in the data that are not explained by the seasonal and trend components. It assumes that the errors follow a white noise process, which means they are independent and identically distributed.

4. Exogenous Variables (X): SARIMAX models incorporate exogenous variables, which are additional factors that can influence the target variable but are not directly modeled as part of the seasonal, trend, or error components. These variables can be categorical or numerical and can have their own lags or relationships with the target variable. Exogenous variables are included in the model through the "exog" parameter.

To use SARIMAX model the data must be stationary, therefor need to use **Augmented Dickey–Fuller Test** to check. The Pvalue result was grater then 0.05 meaning the data is stationary.

The model was trained on the entire data until Feb-2023, validated on the last 3 months of the data - March-2023 - May-2023 and predicted the last 6 months of 2023.

### Results - Question 1.5

The optimal RMSE was not low ( **RMSE:** 475.279) - the minimun value was obtained using:
```
order=(1,1,1), seasonal_order=(1, 1, 1, 12), trend='t', trend_order=2
```

The model itself is not quite accurate, both the rmse and the predicted results show that (one result had a negative value).

The most significant parametes were the addition of the trend - I used "t" as a polinom from the second order, as the trend component in the seasonal decompose for category e can suggest (looks like it's linearly increasing then linearly decreasing which can be assumes as a polimone):
y(t) =a+bt+bt^2

Another significat feature was the order and the seasonal order. I tried multiple combinations to decrease the RMSE but could not find a better configurtion then mentioned above.

**How can we improve the model?**
1. Optimize the order and seasonal order parameters.
2. The model itself should be be retrained more often (every quarter) - the model used was only trained once and predicted 6 months.
3. Infarance - insert the predicted value into the data set and take into considuration for next month prediction.
4. Validation - should validate the model every month and compare to the prediction and make the needed adjustments.
5. Add more exhougenouse variables like credit card usage (i.e market indicator) - this can be used to determine market trends over all - when there is a sense of prosperity - people buy more. This can be factored with a an addition of lag time - say 1 month after an official decleration
6. Additional Models options that can be considured - LSTM, VAR.

#### Sales Predictions

Here are the final model predictions for Product E

| Date | Sales predictions ($) |
| ------ | ------ |
| 2023-06-01 |	414 |
| 2023-07-01 |	621 |
| 2023-08-01 |	556 | 
| 2023-09-01 |	-306 |
| 2023-10-01 |	443 |
| 2023-11-01 |	702 |



## How to run the notebook
First file containing the data preperation and the clean up.

```
utils.ipynb
```

Second file contains the statistical analysis done for the product lines + comparison to competitors - Q1.1

```
stats.ipynb
```

Third file contains the model training, evaluating, predictions and preformance analysis - Q1.2 - Q1.5

```
model.ipynb
```

## Dependencies 
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
from PIL import Image
from statsmodels.tsa.stattools import adfuller
```
