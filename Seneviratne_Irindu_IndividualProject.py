# Load Libraries
from sklearn.linear_model import LinearRegression
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Import data
kickstarter_df = pandas.read_excel(r"C:\Users\irind\Desktop\MMA_Courses\INSY 662\Kickstarter.xlsx")

# Drop canceled, live and suspended states
kickstarter_df= kickstarter_df[kickstarter_df.state != 'canceled']
kickstarter_df= kickstarter_df[kickstarter_df.state != 'live']
kickstarter_df= kickstarter_df[kickstarter_df.state != 'suspended']

#Remove column launch to state change day as it has nan in failure state
kickstarter_df = kickstarter_df.iloc[:,0:44]

#Convert goal to goal_usd
kickstarter_df['goal_usd'] = kickstarter_df['goal']*kickstarter_df['static_usd_rate']

#Remove columns
columns = ['state','project_id','name','pledged',"goal",'disable_communication',"created_at_weekday","created_at_month","created_at_day","created_at_yr","created_at_hr","deadline_month","deadline_day","deadline_yr","deadline_hr","deadline_weekday","spotlight",'state_changed_at','name_len_clean','blurb_len_clean',"state_changed_at_weekday","state_changed_at_month", 'staff_pick',"state_changed_at_day","state_changed_at_yr","state_changed_at_hr","deadline","created_at","launched_at","launch_to_deadline_days"]
kickstarter = kickstarter_df.drop(columns, axis=1)


kickstarter = kickstarter[["usd_pledged", "goal_usd","static_usd_rate", \
                       "country", "currency", \
                       "category", \
                       "name_len", "blurb_len", \
                       "launched_at_weekday",  \
                       "launched_at_month","launched_at_day","launched_at_yr","launched_at_hr","create_to_launch_days"]]

#Dummify variables

kickstarter = pandas.get_dummies(kickstarter, columns = ["country","currency", \
                       "category",\
                       "launched_at_weekday", \
                       "launched_at_month","launched_at_day","launched_at_yr","launched_at_hr"])

# Setup the variables
X = kickstarter.iloc[:,1:]
y = kickstarter["usd_pledged"]

#Standardize X
scaler=StandardScaler()
X_std = scaler.fit_transform(X)

#Using Lasso with Alpha = 2000  to get my predictors

X = kickstarter[['goal_usd', 'name_len', 'country_US', 'category_Flight', 'category_Gadgets', 'category_Hardware', 'category_Sound', 'category_Wearables', 'category_Web', 'launched_at_weekday_Tuesday', 'launched_at_weekday_Wednesday', 'launched_at_yr_2013', 'launched_at_yr_2016', 'launched_at_hr_2', 'launched_at_hr_4', 'launched_at_hr_6', 'launched_at_hr_7', 'launched_at_hr_8', 'launched_at_hr_10']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

#Final model for Regression
rf = RandomForestRegressor(random_state=0, n_estimators=100, max_features=4,max_depth= 8, min_samples_split=20,min_samples_leaf=7)
model = rf.fit(X_train,y_train)

y_test_pred=rf.predict(X_test)
mse = mean_squared_error(y_test,y_test_pred)
print('Training MSE:',mse)


############## 

#Using Kickstarter Grading dataset to test REGRESSION MODEL

kickstarter_grading_df = pandas.read_excel(r"C:\Users\irind\Desktop\MMA_Courses\INSY 662\Kickstarter-Grading.xlsx")

kickstarter_grading_df= kickstarter_grading_df[kickstarter_grading_df.state != 'canceled']
kickstarter_grading_df= kickstarter_grading_df[kickstarter_grading_df.state != 'live']
kickstarter_grading_df= kickstarter_grading_df[kickstarter_grading_df.state != 'suspended']

#Remove column launch to state change day as it has nan in failure state
kickstarter_grading_df = kickstarter_grading_df.iloc[:,0:44]


#Remove columns
kickstarter_grading_df['goal_usd'] = kickstarter_grading_df['goal']*kickstarter_grading_df['static_usd_rate']

columns = ['state','project_id','name','pledged','staff_pick',"goal",'disable_communication',"created_at_weekday","created_at_month","created_at_day","created_at_yr","created_at_hr","deadline_month","deadline_day","deadline_yr","deadline_hr","deadline_weekday","spotlight",'state_changed_at','name_len_clean','blurb_len_clean',"state_changed_at_weekday","state_changed_at_month", "state_changed_at_day","state_changed_at_yr","state_changed_at_hr","deadline","created_at","launched_at","launch_to_deadline_days"]
kickstarter_grading_df = kickstarter_grading_df.drop(columns, axis=1)


kickstarter_grading_df = kickstarter_grading_df[["usd_pledged", "goal_usd","static_usd_rate", \
                       "country", "currency", \
                       "category", \
                       "name_len", "blurb_len", \
                       "launched_at_weekday",  \
                       "launched_at_month","launched_at_day","launched_at_yr","launched_at_hr","create_to_launch_days"]]

#Dummify variables

kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ["country","currency", \
                       "category",\
                       "launched_at_weekday", \
                       "launched_at_month","launched_at_day","launched_at_yr","launched_at_hr"])


#Set Variables 
X_grading = kickstarter_grading_df[['goal_usd', 'name_len', 'country_US', 'category_Flight', 'category_Gadgets', 'category_Hardware', 'category_Sound', 'category_Wearables', 'category_Web', 'launched_at_weekday_Tuesday', 'launched_at_weekday_Wednesday', 'launched_at_yr_2013', 'launched_at_yr_2016', 'launched_at_hr_2', 'launched_at_hr_4', 'launched_at_hr_6', 'launched_at_hr_7', 'launched_at_hr_8', 'launched_at_hr_10']]
y_grading = kickstarter_grading_df["usd_pledged"]


#Using Final Model to predict for Kickstarter grading
y_grading_pred = rf.predict(X_grading)
mse_grading = mean_squared_error(y_grading, y_grading_pred)
print('Testing MSE:',mse_grading)



###################################################################################################

## Classification: Training Model

# Import data
kickstarter_df = pandas.read_excel(r"C:\Users\irind\Desktop\MMA_Courses\INSY 662\Kickstarter.xlsx")

# Drop canceled, live and suspended states
kickstarter_df= kickstarter_df[kickstarter_df.state != 'canceled']
kickstarter_df= kickstarter_df[kickstarter_df.state != 'live']
kickstarter_df= kickstarter_df[kickstarter_df.state != 'suspended']

#Remove column launch to state change day as it has nan in failure state
kickstarter_df = kickstarter_df.iloc[:,0:44]
kickstarter_df = kickstarter_df.dropna()


columns = ['usd_pledged','project_id','name','pledged','disable_communication','state_changed_at',"spotlight",'name_len_clean','blurb_len_clean',"state_changed_at_weekday","state_changed_at_month", "state_changed_at_day","state_changed_at_yr","state_changed_at_hr","deadline","created_at","launched_at","launch_to_deadline_days"]
kickstarter_class = kickstarter_df.drop(columns, axis=1)

kickstarter_class['goal_usd'] = kickstarter_class['goal']*kickstarter_class['static_usd_rate']

kickstarter_class = kickstarter_class[["state", "goal","goal_usd", "static_usd_rate", \
                       "country", "currency", \
                       "staff_pick", "category", \
                       "name_len", "blurb_len","deadline_weekday","created_at_weekday", \
                       "launched_at_weekday", "deadline_month","deadline_day","deadline_yr","deadline_hr", \
                       "created_at_month","created_at_day","created_at_yr", \
                       "created_at_hr","launched_at_month","launched_at_day","launched_at_yr","launched_at_hr","create_to_launch_days"]]

#Dummify variables
kickstarter_class = pandas.get_dummies(kickstarter_class, columns = ["state","country","currency", \
                       "staff_pick", "category", \
                       "deadline_weekday","created_at_weekday", \
                       "launched_at_weekday", "deadline_month","deadline_day","deadline_yr","deadline_hr", \
                       "created_at_month","created_at_day","created_at_yr", \
                       "created_at_hr","launched_at_month","launched_at_day","launched_at_yr","launched_at_hr"])
        
#Setting up variables       
y=kickstarter_class['state_successful']
X= kickstarter_class.drop(['state_successful','state_failed'],axis=1)

#Variables after feature selection
X = kickstarter_class[['goal_usd','static_usd_rate','blurb_len','name_len','create_to_launch_days','staff_pick_False','category_Plays','category_Software','category_Web']]


#Classification training model
randomforest= RandomForestClassifier(random_state=0)
model = randomforest.fit(X,y)



#Standardize variables
standardizer = StandardScaler()
X_std= standardizer.fit_transform(X)

#Split
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.3,random_state=5)

## Classification training model
knn_final = KNeighborsClassifier(n_neighbors = 14).fit(X_train,y_train)
y_test_pred = knn_final.predict(X_test)
print("Accuracy Score for Training Model:",accuracy_score(y_test,y_test_pred))


################

#Using Kickstarter Grading on classification model

kickstarter_grading_df = pandas.read_excel(r"C:\Users\irind\Desktop\MMA_Courses\INSY 662\Kickstarter-Grading.xlsx")

# Drop canceled, live and suspended states
kickstarter_grading_df= kickstarter_grading_df[kickstarter_grading_df.state != 'canceled']
kickstarter_grading_df= kickstarter_grading_df[kickstarter_grading_df.state != 'live']
kickstarter_grading_df= kickstarter_grading_df[kickstarter_grading_df.state != 'suspended']

#Remove column launch to state change day as it has nan in failure state
kickstarter_grading_df = kickstarter_grading_df.iloc[:,0:44]
kickstarter_grading_df = kickstarter_grading_df.dropna()


columns = ['usd_pledged','project_id','name','pledged','disable_communication','state_changed_at',"spotlight",'name_len_clean','blurb_len_clean',"state_changed_at_weekday","state_changed_at_month", "state_changed_at_day","state_changed_at_yr","state_changed_at_hr","deadline","created_at","launched_at","launch_to_deadline_days"]
kickstarter_grading_df = kickstarter_grading_df.drop(columns, axis=1)

kickstarter_grading_df['goal_usd'] = kickstarter_grading_df['goal']*kickstarter_grading_df['static_usd_rate']

kickstarter_grading_df = kickstarter_grading_df[["state", "goal","goal_usd", "static_usd_rate", \
                       "country", "currency", \
                       "staff_pick", "category", \
                       "name_len", "blurb_len","deadline_weekday","created_at_weekday", \
                       "launched_at_weekday", "deadline_month","deadline_day","deadline_yr","deadline_hr", \
                       "created_at_month","created_at_day","created_at_yr", \
                       "created_at_hr","launched_at_month","launched_at_day","launched_at_yr","launched_at_hr","create_to_launch_days"]]

#Dummify variables
kickstarter_grading_df = pandas.get_dummies(kickstarter_grading_df, columns = ["state","country","currency", \
                       "staff_pick", "category", \
                       "deadline_weekday","created_at_weekday", \
                       "launched_at_weekday", "deadline_month","deadline_day","deadline_yr","deadline_hr", \
                       "created_at_month","created_at_day","created_at_yr", \
                       "created_at_hr","launched_at_month","launched_at_day","launched_at_yr","launched_at_hr"])

# usd_pledged should be removed
X_grading = kickstarter_grading_df[['goal_usd','static_usd_rate','blurb_len','name_len','create_to_launch_days','staff_pick_False','category_Plays','category_Software','category_Web']]
y_grading = kickstarter_grading_df["state_successful"]


from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std= standardizer.fit_transform(X_grading)


#Model
knn_final = KNeighborsClassifier(n_neighbors = 18).fit(X_train,y_train)
y_test_pred = knn_final.predict(X_grading)
print('Accuracy score for Testing Model:',accuracy_score(y_grading,y_test_pred))




