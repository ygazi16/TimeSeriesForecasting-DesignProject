from graph import deneme
from Univariate_forecasting import forecast_algorithm, graphVal


data = deneme()



labels = [row[0] for row in data]
values = [row[1] for row in data]

#print(labels)

#a = forecast_algorithm("xgb",50)
#print(a)

b = graphVal()
print(b)


