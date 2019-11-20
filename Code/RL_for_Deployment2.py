import numpy as np
import pandas as pd
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import timedelta
import itertools
import holidays
us_holidays = holidays.UnitedStates()
import pickle
import io
import base64
import scipy.optimize as opt 
np.set_printoptions(precision=2)
# ### Input required files and models
# 1. xg_reg => Predict price
# 2. model_xgb_class => Predict whether demand > 0
# 3. xg_reg_dem_above0 => Predict demand based on price and other factors
# 4. df_dem_for_reg => For past demand data

xg_reg = pickle.load(open("../models/price_reg.pkl", 'rb'))
model_xgb_class = pickle.load(open("../models/demand_classifier.pkl", 'rb'))
xg_reg_dem_above0 = pickle.load(open("../models/xg_reg_dem_above0.pkl", 'rb'))
df_req_price = pickle.load(open("../models/past_data_price.pkl", 'rb'))
df_dem_for_reg = pickle.load(open("../models/past_data_demand.pkl", 'rb'))


# #### Calculate few things for writing functions
# 1. Number of columns in price model
# 2. Number of columns in demand model

price_x = pd.get_dummies(df_req_price, columns=['Agent', 'Meal', 'ArrivalDateMonth', 'AssignedRoomType'])
price_x = price_x.drop(['ADR'],axis = 1)
dem_x = pd.get_dummies(df_dem_for_reg, columns=['Agent', 'AssignedRoomType', 'Meal', 'ArrivalDateMonth'])
dem_x = dem_x.drop(['date','demand'],axis = 1)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_daterange(start_date, end_date, step):
    while start_date <= end_date:
        yield start_date
        start_date += step

def price_fun(df_pred):
    df_pred_2 = pd.get_dummies(df_pred, columns=['Agent', 'Meal', 'ArrivalDateMonth', 'AssignedRoomType'])     
    #Create a dataframe of zeros with rows = no: of input, col = standard no: of col
    zero_data = np.zeros(shape=(df_pred.shape[0],price_x.shape[1]))
    X_fun = pd.DataFrame(zero_data, columns=price_x.columns)
    
    for i in df_pred_2.columns:
        X_fun[i] = df_pred_2[i]
    
    return(xg_reg.predict(X_fun))

def min_max(a): 
  minpos = a.index(min(a)) 
  maxpos = a.index(max(a)) 
  return(minpos,maxpos)
 
#Function that is useful for directly feeding in the values
def price_single_fun(#Country,
    ArrivalDateYear, ArrivalDateMonth, ArrivalDateWeekNumber,
                     Agent,#DistributionChannel,
                     AssignedRoomType, Meal,
                     StaysInWeekendNights=0,StaysInWeekNights=0,Adults =2, Children=0, Babies=0):
    
    temp = df_req_price.loc[:, df_req_price.columns != 'ADR']
    out_single = pd.DataFrame(index=range(1),columns=temp.columns)
    #out_single.loc[0,"Country"] = Country
    out_single.loc[0,"ArrivalDateYear"] = ArrivalDateYear
    out_single.loc[0,"ArrivalDateMonth"] = ArrivalDateMonth
    out_single.loc[0,"ArrivalDateWeekNumber"] = ArrivalDateWeekNumber
    out_single.loc[0,"StaysInWeekendNights"] = StaysInWeekendNights
    out_single.loc[0,"StaysInWeekNights"] = StaysInWeekNights
    out_single.loc[0,"Adults"] = Adults
    out_single.loc[0,"Children"] = Children
    out_single.loc[0,"Babies"] = Babies
    out_single.loc[0,"AssignedRoomType"] = AssignedRoomType
    out_single.loc[0,"Meal"] = Meal
    out_single.loc[0,"Agent"] = Agent
    #out_single.loc[0,"DistributionChannel"] = DistributionChannel
    
    for x in out_single.columns:
        out_single[x]=out_single[x].astype(temp[x].dtypes.name)

    
    return(price_fun(out_single))

def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

def dem_fun(df_dem_pred):
    df_dem_pred_2 = pd.get_dummies(df_dem_pred, columns=['Agent', 'AssignedRoomType', 'Meal', 'ArrivalDateMonth'])      
    
    #Create a dataframe of zeros with rows = no: of input, col = standard no: of col
    zero_data = np.zeros(shape=(df_dem_pred.shape[0],dem_x.shape[1]))
    X_fun = pd.DataFrame(zero_data, columns=dem_x.columns)
    
    for j in df_dem_pred_2.columns:
        X_fun[j] = df_dem_pred_2[j]
    
    #Take anything <0 as 0
    xg_dem = np.floor(model_xgb_class.predict(X_fun) * xg_reg_dem_above0.predict(X_fun)) #XGBoost prediction
    temp123 = np.array(xg_dem)
    temp123[temp123 <0] = 0
    
    output_demand = list(temp123)

    
    return(output_demand)

#Function that is useful for directly feeding in the values
def dem_single_fun(Agent, AssignedRoomType, Meal,ArrivalDateMonth,ADR,Week_day, 
                        ArrivalDateWeekNumber,ArrivalDateYear,Day,public_holiday):
    
    temp = df_dem_for_reg.loc[:, df_dem_for_reg.columns != 'demand']
    temp = temp.loc[:, temp.columns != 'date']
    out_single = pd.DataFrame(index=range(1),columns=temp.columns)
    #out_single.loc[0,"Country"] = Country
    out_single.loc[0,"Agent"] = Agent
    out_single.loc[0,"AssignedRoomType"] = AssignedRoomType
    out_single.loc[0,"Meal"] = Meal
    out_single.loc[0,"ArrivalDateMonth"] = ArrivalDateMonth
    out_single.loc[0,"ADR"] = ADR
    out_single.loc[0,"Week_day"] = Week_day
    out_single.loc[0,"ArrivalDateWeekNumber"] = ArrivalDateWeekNumber
    out_single.loc[0,"ArrivalDateYear"] = ArrivalDateYear
    out_single.loc[0,"Day"] = Day
    out_single.loc[0,"public_holiday"] = public_holiday
    for x in out_single.columns:
        out_single[x]=out_single[x].astype(temp[x].dtypes.name)

    
    return(dem_fun(out_single))   

# Find the optimal distribution of prices (price probabilities) given fixed price levels, 
# corresponding demand levels, and availbale product inventory.
# 
# Inputs:
#   prices, demands, and revenues are vectors (i-th element corresponds to i-th price level)
#   inventory is a scalar (number of availbale units)
def optimal_price_probabilities(prices, demands):   
    revenues = np.multiply(prices, demands)
    
    L = len(prices)
    M = np.full([1, L], 1)
    B = [[1]]
    Df = [demands]
    print("demand",demands)
    print("price",prices)
    print(-np.array(revenues).flatten())
    res = linprog(-np.array(revenues).flatten(), 
                  # The coefficients of the linear objective function to be minimized.
                  # Flattened, cos 1D array is required
                  A_eq=M,
                  #The equality constraint matrix. Each row of ``A_eq`` specifies the
                  #coefficients of a linear equality constraint on ``x``.
                  
                  b_eq=B,
                  #The equality constraint vector. Each element of ``A_eq @ x`` must equal
                  #the corresponding element of ``b_eq``.
                  
                  A_ub=Df, 
                  #The inequality constraint matrix. Each row of ``A_ub`` specifies the
                  #coefficients of a linear inequality constraint on ``x``
                  
                  b_ub=np.array([1000]), #Taking some arbitrary inventory 
                  #The inequality constraint vector. Each element represents an
                  #upper bound on the corresponding value of ``A_ub @ x``.
                  
                  #A_ub @ x <= b_ub
                  #A_eq @ x == b_eq
                  #lb <= x <= ub
                  
                  bounds=(0, None))
    price_prob = np.array(res.x).reshape(1, L).flatten()
    # res. x = The values of the decision variables that minimizes the objective function while satisfying the constraints.
    
    # -Res.fun gives the maximum revenue
    return -res.fun, price_prob

np.random.seed(42)    
#Prior distribution
def gamma(alpha, beta): 
    shape = alpha
    scale = 1/beta
    return np.random.gamma(shape, scale)

def sample_demands_from_model(θ):
    return list(map(lambda v: gamma(v['mean'], 1), θ)) #Hard coding beta = 1, alpha is mean !
    #return list(map(lambda v: gamma(v['alpha'], v['beta']), θ))










#Pending , need to plot date in x axis
def visualize_snapshot(st,en,history):
   
    t = (en - st).days - 2
    st = dt.datetime.strptime(str(st), "%Y-%m-%d")
    en = dt.datetime.strptime(str(en), "%Y-%m-%d")
    len1 = history.shape[0]
    
    date_range = list(get_daterange(en, st,timedelta(days=1)))
    img = io.BytesIO()
    fig, ax =plt.subplots(6,1,figsize=(16,25))
    # fig.subplots_adjust(hspace=0.5)
    sns.lineplot(range(history.shape[0]),history.act_price, ax=ax[0])
    sns.barplot(list(range(history.shape[0])),history.act_demand, ax=ax[1])
    sns.barplot(list(range(history.shape[0])),history.act_Revenue, ax=ax[2])
    sns.lineplot(range(history.shape[0]),history.selected_price, ax=ax[3])
    sns.barplot(list(range(history.shape[0])),history.obs_demand, ax=ax[4])
    sns.barplot(list(range(history.shape[0])),history.Revenue, ax=ax[5])
    # fig.show()
    fig.savefig(img,format='png')
    img.seek(0)
    for i in range(0,6):
        ax[i].set_facecolor("#c0c0c0")
    graph_url = base64.b64encode(img.getvalue()).decode()
    
    return 'data:image/png;base64,{}'.format(graph_url)
temp_df1 = pd.DataFrame(df_dem_for_reg.values)
temp_df1.columns = df_dem_for_reg.columns
temp_df1["code"] = temp_df1['Agent'].astype(str) + temp_df1['AssignedRoomType'].astype(str) + temp_df1['Meal'].astype(str)
temp_df1["code"].value_counts()[:10]


def RL_fun(Agent1 = ["7","14","28"],AssignedRoomType1 = ["A"],Meal1 = ["BB"],
           start_date = dt.date(2017, 7, 1),end_date = dt.date(2017, 8, 1), price_space = 5,
           price_explore = 20,initial_rampup = 1.0,ploton =1,sel_price_on = 0,dem_input = -1,history_full = [],
          monthDict={1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
           7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}):    
    #Price space is the price range interval we will try
    #Price explore is the price explore above and below curr price. Curr price is given by price fun
    #dem_input is for user to enter a demand at and observed price
    myresult={}
    input_df=pd.DataFrame()
    input_df = pd.DataFrame(expandgrid(Agent1,AssignedRoomType1,Meal1))
    input_df.columns = ["Agent","AssignedRoomType","Meal"]
    input_df['code'] = input_df['Agent'].astype(str) + input_df['AssignedRoomType'].astype(str) + input_df['Meal'].astype(str)

    np.random.seed(42)
    user_given_history = 0
    if(len(history_full) != 0):
        user_given_history = 1
    
    for i in range(input_df.shape[0]): 
        print(input_df['code'][i])
        history = []
        θ = []
        act_history = []
        
        code = str(input_df.code[i])

        #Find a price for the particular demand
        #First find out the date related features
        curr_date = start_date
        curr_year = curr_date.year
        curr_month = monthDict[curr_date.month]
        curr_week = curr_date.isocalendar()[1]
        curr_weekday = curr_date.isoweekday()
        curr_day = curr_date.day
        curr_holiday = (start_date in us_holidays) * 1

        #Second find other features from the df element (type of product)
        curr_Agent = input_df.Agent[i]
        curr_AssignedRoomType = input_df.AssignedRoomType[i]
        curr_Meal = input_df.Meal[i]
        

        # =>Check from history whether already the code is run before 
        # a. Find the count of the code in the history
        if((user_given_history == 1)):
            for l in range(len(history_full)):
                if(history_full[l][1] == code):
                    θ = history_full[l][-1] #Latest price and hyperparameter list - because we run loop for len
        #print("Theta",θ)
            prices =[]
            for m in range(len(θ)):
                prices.append(θ[m]['price']) 
            
                
            #Old logic not working
            #num = (np.pad(np.array(history_full)[:, 0], (0, 0), 'constant'))
            #num = list(num)
            #prv_count = num.count(code)

        #Old logic not working
        # b. Assign previous theta (most latest) if previously availble
        #if(prv_count > 0):
            #θ = list(np.pad(np.array(history_full)[prv_count-1,4], (0, 0), 'constant'))
            #print(θ)
        # b. Else initalize
        else:
            #Now find the predicted price and make a price list
            pred_price = price_single_fun(curr_year,curr_month,curr_week,curr_Agent,curr_AssignedRoomType,curr_Meal)
            
            #prices = list(range(int(curr_df['ADR'].min()-price_explore),int(curr_df['ADR'].max()+price_explore),price_space))
            
            prices = list(range((int(pred_price) - price_explore),(int(pred_price) + price_explore),price_space))
            prices = [p for p in prices if p > 0]
            
            #For each price keep a demand distribution gamma distribution with
            #alpha = avg demand on that day at various prices
            #beta = 1
            # mean = demand
            demand_at_price = [0] * len(prices)
            temp_count = 0
            temp_dem = 0
            for p in prices:
                #Initalize with model demands
                demand_at_price[temp_count] = dem_single_fun(curr_Agent, curr_AssignedRoomType, curr_Meal,curr_month,p,
                                             curr_weekday,curr_week,curr_year,curr_day,curr_holiday)
                demand_at_price[temp_count] = demand_at_price[temp_count][0] * initial_rampup
                
                #To eliminate below zeros values
                #demand_at_price[temp_count] = max(0,demand_at_price[temp_count])
                #print(demand_at_price[temp_count])
                
                demand_at_price[temp_count] = np.random.poisson(demand_at_price[temp_count], 1)[0]
                θ.append({'price': p, 'alpha': demand_at_price[temp_count], 'beta': 1.00, 'mean': demand_at_price[temp_count]})
                temp_dem += demand_at_price[temp_count]
                temp_count+=1
                
            #Second loop to update the alpha and mean - For keeping all demand same not using now
            #temp_dem_mean = temp_dem/len(prices)
            #for p in prices:
                #print(demand_at_price[temp_count])
                #θ.append({'price': p, 'alpha': temp_dem_mean, 'beta': 1.00, 'mean': temp_dem_mean})
                
        
        T = (end_date - start_date).days
        date_range = [0] * T

        for t in range(0, T):   # simulation loop
            
            date_range[t] = curr_date
            #For actual Revenue calculation
            temp99_df = pd.DataFrame(df_dem_for_reg.values)
            temp99_df.columns = df_dem_for_reg.columns
            temp99_df["code_date"] = df_dem_for_reg['Agent'].astype(str) + df_dem_for_reg['AssignedRoomType'].astype(str) + df_dem_for_reg['Meal'].astype(str) + df_dem_for_reg['date'].astype(str)
            
            code_date = code + str(curr_date)
            
            #print("code_date",code_date)
            #print(temp99_df.code_date)
            
            temp100_df = temp99_df[temp99_df.code_date == code_date]
            
                                         
            temp100_df.columns = temp99_df.columns
            temp100_df.index = range(temp100_df.shape[0])
            
            if(temp100_df.empty):
                act_price = 0
                act_demand = 0
                act_Revenue = 0
                
            else:
                act_price = temp100_df.ADR[0]
                act_demand = temp100_df.demand[0]
                act_Revenue = act_price * act_demand
            
            act_history.append([act_price, act_demand, act_Revenue])
                                    
            demands = sample_demands_from_model(θ)
            #print("Theta",θ)
            #demands = sample_demands_from_model(θ)[-len(prices):] #Doing this as appending problem keeps arising
            demands = np.round(np.array(demands))
            #print("demands1",demands)
            
            #demands = np.round(sample_demands_from_model(θ))
            demands[demands <=0] = 0
            
            #demands = demands[demands >=0]

            #Demand is a gamma distribution with alpha and beta values
            #print(tabulate(np.array(θ), tablefmt="fancy_grid"))
            
            #print("demands = ", np.array(demands))
            #print("prices = ", np.array(prices))
            
            pred_revenue, price_probs = optimal_price_probabilities(prices, demands)
            
            #print("revenue = ", np.array(pred_revenue))
            #print("price probs = ", np.array(price_probs))
            #print("history full",history_full)
                     
            # select one best price
            #price_index_t = list(price_probs).index(max(price_probs))
            price_index_t = min_max(list(price_probs))[1]
            #Not using below line - old code
            #price_index_t = np.random.choice(len(prices), 1, p=price_probs)[0]
            
            #Finding highest revenue - Toggling between this and the price probs
            #revenue_array = np.array(prices) * np.array(demands)
            #price_index_t = np.argmax(revenue_array)
            
            #print(price_index_t)
            
            #print("price_index_t",price_index_t)
            price_t = prices[price_index_t]
            
            if(sel_price_on == 1):
                print("Selected Price",price_t)

            #Not using below
            #pred_demand = pred_revenue/price_t
            #print('selected price %.2f => predicted demand %.2f, predicted revenue %.2f' % (price_t, pred_demand, pred_revenue))

            # sell at the selected price and observe demand - actual
            dem_for_poisson = dem_single_fun(curr_Agent, curr_AssignedRoomType, curr_Meal,curr_month,price_t,
                                             curr_weekday,curr_week,curr_year,curr_day,curr_holiday)
            #demand_t = dem_for_poisson
            #print("t",t)
            if(dem_input == -2):  #No need input the last time and hence second condition
                your_dem = input("Enter the actual demand: ")
                dem_input2 = int(your_dem)
                demand_t = np.random.poisson(dem_input2, 1)[0] #Taking demand as a poisson dist of user input
                #print("demand_t",demand_t)
            else:
                demand_t = np.random.poisson(dem_for_poisson, 1)[0]
            
            #print(demand_t)
            #print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t, demand_t, demand_t*price_t))

            # update model parameters
            #print(tabulate(np.array(θ), tablefmt="fancy_grid"))
            #v = θ[price_index_t]
            #print(f"Alpha is {v['alpha']}.") 
            θ[price_index_t]['alpha'] = θ[price_index_t]['alpha'] + demand_t
            θ[price_index_t]['beta'] = θ[price_index_t]['beta'] + 1
            θ[price_index_t]['mean'] = θ[price_index_t]['alpha'] / θ[price_index_t]['beta']
            #v['alpha'] = v['mean'] #Making alpha = mean !!! New technique !
            #v['beta'] = 1 #Making beta same so as to get the demand with highest mean next time !
            
            theta_trace = []
            for v in θ:
                theta_trace.append(v.copy())
            history.append([price_t, demand_t, demand_t*price_t, theta_trace])
            #print("price_t",price_t)
            history_full.append([curr_date,code,act_price, act_demand, act_Revenue,price_t, demand_t, demand_t*price_t, theta_trace])

                  
            #First find out the date related features
            curr_date += timedelta(days=1)
            curr_year = curr_date.year
            curr_month = monthDict[curr_date.month]
            curr_week = curr_date.isocalendar()[1]
            curr_weekday = curr_date.isoweekday()
            curr_day = curr_date.day
            curr_holiday = (curr_date in us_holidays) * 1
             
        temp5_df = pd.DataFrame(history)
        temp5_df.columns = ["selected_price","obs_demand","Revenue","params"]
        temp6_df = pd.DataFrame(act_history)
        temp6_df.columns = ["act_price","act_demand","act_Revenue"]
        temp7_df = pd.concat([temp6_df, temp5_df], axis=1)
        #print(code)
        #print(temp5_df.selected_price.value_counts())
        #fig = plt.figure(figsize = (10, 12))
        myfig_url=""
        if(ploton == 1):
            plt.subplots_adjust(hspace = 0.5)
            myfig_url=visualize_snapshot(start_date,end_date,temp7_df)   # visualize the final state of the simulation
            myresult[code]=myfig_url
    
   
    history_df = pd.DataFrame(history_full)
    #print(history_full)
    #history_df.index = date_range
    history_df.columns = ["date","code","act_price","act_demand","act_Revenue","selected_price","obs_demand","Revenue","params"]
    # new_df = history_df.groupby("code", as_index=False).sum()[['code','act_Revenue',"Revenue"]]
    # row = ["total", sum(new_df.act_Revenue), sum(new_df.Revenue)]
    # new_df.loc[len(new_df)] = row
    return history_df,history_full,myresult




def getResults(Agent1 = ["9","7","14"],AssignedRoomType1 = ["A"],Meal1 = ["BB"],start_date = dt.date(2017, 7, 1)):
    # print(str(start_date))

    # date_1 = dt.datetime.strptime(str(start_date), "%yyyy-%mm-%dd")
    end_date = (start_date+timedelta(days=15)).isoformat() 
    year = end_date[0:4]
    month = end_date[5:7]
    day =end_date[8:10]
    # print(mydict)
    end_date = dt.date(int(year),int(month),int(day))
    print("sad",type(end_date))
    a, b , myresult = RL_fun(Agent1=Agent1 , AssignedRoomType1=AssignedRoomType1 , Meal1=Meal1,start_date=start_date,end_date=end_date)
    new_df = a.groupby("code", as_index=False)[['code','act_Revenue',"Revenue"]].sum()
    row = ["total", sum(new_df.act_Revenue), sum(new_df.Revenue)]
    new_df.loc[len(new_df)] = row
    new_df
    print(a.act_Revenue.sum())
    print(a.Revenue.sum())
    fig, ax1 = plt.subplots(figsize=(10, 5))
    tidy = new_df.melt(id_vars='code').rename(columns=str.title)
    ax1.set_facecolor("#f0f0f0")
    img = io.BytesIO()
    sns.barplot(x='Code', y="Value",hue = 'Variable', data=tidy, ax=ax1)
    sns.despine(fig)
    fig.savefig(img,format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    graph_url_final =  'data:image/png;base64,{}'.format(graph_url)
    return a,b, myresult ,new_df, graph_url_final


def RL_fun3(start_date3 = dt.date(2017, 7, 1), price_space3= 5, price_explore3 = 10,n_play3 = 4):
# prior distribution for each price - gamma(α, β)
    θ3 = []
    #For each price keep a demand distribution gamma distribution with alpha = 30, beta = 1 , mean = 30

    strt_p3 = input("Give a start price :")
    strt_p3 = float(strt_p3)
    strt_dmd3 = input("Give a start demand for the product :")
    strt_dmd3 = int(strt_dmd3)

    prices3 = list(range((int(strt_p3) - price_explore3),(int(strt_p3) + price_explore3 + price_space3),price_space3))

    curr_date3 = start_date3

    for p in prices3:
        θ3.append({'price': p, 'alpha': strt_dmd3, 'beta': 1.00, 'mean': strt_dmd3})

    history3 = []
    selected_prices3 = []
    selected_dmd3 = []
    
    for t in range(0, n_play3):              # simulation loop
        print("Date :",curr_date3)
        demands3 = sample_demands_from_model(θ3)

        pred_revenue3, price_probs3 = optimal_price_probabilities(prices3, demands3)

        # select one best price
        price_index_t3 = np.random.choice(len(prices3), 1, p=price_probs3)[0]
        price_t3 = prices3[price_index_t3]

        print("Selected price as per algorithm" , price_t3)

        print("Price Ranges", prices3)
        act_price3 = input("Enter your price selection: ")
        act_price3 = float(act_price3)
        
        while(act_price3 not in prices3):
            print("Please select price only from choices ",act_price3)
            act_price3 = input("Enter your price selection: ")                    
            act_price3 = float(act_price3)
        

        price_index_t3 = prices3.index(int(act_price3))

        price_t3 = act_price3
        
        selected_prices3.append(price_t3)
        

        #Not using below
        #pred_demand = pred_revenue/price_t
        #print('selected price %.2f => predicted demand %.2f, predicted revenue %.2f' % (price_t, pred_demand, pred_revenue))

        # sell at the selected price and observe demand - actual
        act_dmd3 = input("Enter the actual demand :")
        act_dmd3 = float(act_dmd3)
        demand_t3 = act_dmd3
        
        selected_dmd3.append(demand_t3)
        
        print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t3, demand_t3, demand_t3*price_t3))

        theta_trace3 = []
        for v in θ3:
            theta_trace3.append(v.copy())
        history3.append([price_t3, demand_t3, demand_t3*price_t3, theta_trace3])

        # update model parameters
        v = θ3[price_index_t3] 
        v['alpha'] = v['alpha'] + demand_t3
        v['beta'] = v['beta'] + 1
        v['mean'] = v['alpha'] / v['beta']
        
        curr_date3 = curr_date3 + timedelta(1)
        #print("history",pd.DataFrame(history))
        print("")
        return selected_prices3 , selected_dmd3
        



def getResults2(start_date3 = dt.date(2017, 7, 1), price_space3= 5, price_explore3 = 10,n_play3 = 4):
    selected_prices3,selected_dmd3=RL_fun3(start_date3,price_space3,price_explore3,n_play3)
    img = io.BytesIO()
    fig, ax =plt.subplots(3,1)
    sns.lineplot(list(range(len(selected_prices3))),selected_prices3, ax=ax[0])
    sns.barplot(list(range(len(selected_prices3))),selected_dmd3, ax=ax[1])
    sns.barplot(list(range(len(selected_prices3))),np.array(selected_prices3) * np.array(selected_dmd3), ax=ax[2])
    fig.savefig(img,format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(graph_url)











def createDemoResult(price_t3,prices3,history3,selected_prices3,selected_dmd3,θ3,curr_date3 = dt.date(2017, 7, 1),next_price=1,next_demand=1, price_space3= 5, price_explore3 = 10,):
        print("Date :",curr_date3)
        demands3 = sample_demands_from_model(θ3)

        pred_revenue3, price_probs3 = optimal_price_probabilities(prices3, demands3)

        # select one best price
        

#         print("Selected price as per algorithm" , price_t3)

#         print("Price Ranges", prices3)
#         act_price3 = input("Enter your price selection: ")
        act_price3 = float(next_price)
        
        price_index_t3 = prices3.index(int(act_price3))

        price_t3 = act_price3
        
        selected_prices3.append(price_t3)
        

        #Not using below
        #pred_demand = pred_revenue/price_t
        #print('selected price %.2f => predicted demand %.2f, predicted revenue %.2f' % (price_t, pred_demand, pred_revenue))

        # sell at the selected price and observe demand - actual
#         act_dmd3 = input("Enter the actual demand :")
        act_dmd3 = float(next_demand)
        demand_t3 = act_dmd3
        
        selected_dmd3.append(demand_t3)
        
        print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t3, demand_t3, demand_t3*price_t3))

        theta_trace3 = []
        for v in θ3:
            theta_trace3.append(v.copy())
        history3.append([price_t3, demand_t3, demand_t3*price_t3, theta_trace3])

        # update model parameters
        v = θ3[price_index_t3] 
        v['alpha'] = v['alpha'] + demand_t3
        v['beta'] = v['beta'] + 1
        v['mean'] = v['alpha'] / v['beta']
        
        curr_date3 = curr_date3 + timedelta(1)
        #print("history",pd.DataFrame(history))
        print("")
        price_index_t3 = np.random.choice(len(prices3), 1, p=price_probs3)[0]
        price_t3 = prices3[price_index_t3]
        return price_t3


def begindemoprocess(start_date3 = dt.date(2017, 7, 1),start_price=1,start_demand=1, price_space3= 5, price_explore3 = 10):
    strt_p3 = float(start_price)
    strt_dmd3 = int(start_demand)

    prices3 = list(range((int(strt_p3) - price_explore3),(int(strt_p3) + price_explore3 + price_space3),price_space3))
    θ3=[]
    curr_date3 = start_date3

    for p in prices3:
        θ3.append({'price': p, 'alpha': strt_dmd3, 'beta': 1.00, 'mean': strt_dmd3})
    demands3 = sample_demands_from_model(θ3)
    pred_revenue3, price_probs3 = optimal_price_probabilities(prices3, demands3)
    # select one best price
    price_index_t3 = np.random.choice(len(prices3), 1, p=price_probs3)[0]
    price_t3 = prices3[price_index_t3]
    print("Selected price as per algorithm" , price_t3)
    print("Price Ranges", prices3)
    history3 = []
    selected_prices3 = []
    selected_dmd3 = []
    return price_t3,prices3,history3,selected_prices3,selected_dmd3,θ3,curr_date3


def showGraph(selected_prices3,selected_dmd3):
    img = io.BytesIO()
    fig, ax =plt.subplots(3,1,figsize=(16,8))
    fig.subplots_adjust(hspace=0.5)
    sns.lineplot(list(range(len(selected_prices3))),selected_prices3, ax=ax[0])
    ax[0].set_title('Price Selected')
    sns.barplot(list(range(len(selected_prices3))),selected_dmd3, ax=ax[1],)
    ax[1].set_title('Demand')
    sns.barplot(list(range(len(selected_prices3))),np.array(selected_prices3) * np.array(selected_dmd3), ax=ax[2])
    ax[2].set_title('Revenue')
    fig.savefig(img,format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(graph_url)
    

