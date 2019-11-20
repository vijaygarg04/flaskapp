import flask
from flask import request
import pandas as pd
import pickle
from flask import Flask
import datetime as dt
import RL_for_Deployment2 as rl
from werkzeug.datastructures import ImmutableMultiDict
from datetime import timedelta


app = Flask(__name__)

df_req_price = pickle.load(open("past_data_demand.pkl", 'rb'))
dataset = pd.read_csv('dummydata1.csv')
agents = df_req_price['Agent'].unique()
meals =  df_req_price['Meal'].unique()
rooms = df_req_price['AssignedRoomType'].unique()
agents_len =  len (agents)
meals_len = len(meals)
rooms_len = len(rooms)
price_t3=1
prices3=1
history3=1
selected_prices3=1
selected_dmd3=1
θ3=1
curr_date3=1
next_price=1
next_demand=1

monthlist=['','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# print(agents)
# print(meals)
# print(rooms)

# Initialize the app
app = flask.Flask(__name__)
@app.route("/demo",methods=["GET","POST"])
def fun():
    print('Demo')
    global price_t3 , prices3,history3,selected_prices3,selected_dmd3,θ3,curr_date3,next_price,next_demand
    if(request.args):
        print('hi11',request.args)
        
        print('request',request)
        imd = ImmutableMultiDict(request.args)
        mydict=imd.to_dict(flat=False)
        
        print(len(mydict))
        print(mydict['submit_button'])
        if((mydict['submit_button'][0])=='Begin Demo Process'):
             
            startprice = mydict['startprice']
            startprice=int(startprice[0])
            startdate = mydict['startdate']
            startdate = startdate[0]
            year = startdate[0:4]
            month = startdate[5:7]
            day =startdate[8:10]
            # print(mydict)
            startdate = dt.date(int(year),int(month),int(day))
            startdemand =mydict['startdemand']
            startdemand=int(startdemand[0])
            pricespace = mydict['pricespace']
            pricespace=int(pricespace[0])
            priceexplore = mydict['priceexplore']
            priceexplore=int(priceexplore[0])
            mydate1 = year +"-"+ monthlist[int(month)]+"-"+day
            price_t3,prices3,history3,selected_prices3,selected_dmd3,θ3,curr_date3=rl.begindemoprocess(startdate,startprice,startdemand,pricespace,priceexplore)
            print('pricet3',price_t3)
            print('prices3',prices3)
            print('history3',history3)
            print('selected_prices3',selected_prices3)
            print('selected_dmd3',selected_dmd3)
            print('θ3',θ3)
            print('curr_date3',curr_date3)
            
            return flask.render_template('code.html',
                                        prices=prices3,
                                        price_t3=price_t3,
                                        curr_date=mydate1)

        elif((mydict['submit_button'][0])=='process next'):
            
            next_price = (mydict['nextprice'])[0]
            next_demand =(mydict['nextdemand'])[0]
            print('pricet3',price_t3)
            print('prices3',prices3)
            print('history3',history3)
            print('selected_prices3',selected_prices3)
            print('selected_dmd3',selected_dmd3)
            print('θ3',θ3)
            print('curr_date3',curr_date3)
            print('next_price',next_price)
            print('next_demand',next_demand)
            print()
            print()
            curr_date3 = str(curr_date3)
            year = curr_date3[0:4]
            month = curr_date3[5:7]
            day =curr_date3[8:10]
            mydate1 = year +"-"+ monthlist[int(month)]+"-"+day
            # print(mydict)
            curr_date3 = dt.date(int(year),int(month),int(day))
            curr_date3 = curr_date3 +timedelta(1)
            price_t3 = rl.createDemoResult(price_t3,prices3,history3,selected_prices3,selected_dmd3,θ3,curr_date3,next_price,next_demand)
            print(price_t3)
            return flask.render_template('code.html',
                                        prices=prices3,
                                        price_t3=price_t3,
                                        curr_date=mydate1)

        else:
            print('hi123hj')
            curr_date3 = str(curr_date3)
            year = curr_date3[0:4]
            month = curr_date3[5:7]
            day =curr_date3[8:10]
            graphurl = rl.showGraph(selected_prices3,selected_dmd3)
            mydate1 = year +"-"+ monthlist[int(month)]+"-"+day
            return flask.render_template('code.html',
                                        prices=prices3,
                                        price_t3=price_t3,
                                        graphurl=graphurl,
                                        curr_date=mydate1)

        
        
    else :
        print('hii1i1i22')
        return  flask.render_template('code.html',
                                        )

@app.route("/history",methods=["GET","POST"])
@app.route("/", methods=["GET","POST"]) 
def predict():
    print('hi',request.args)
    if(request.args):
        
        print(request.args)

        # print(len(request.args))
        # print(type(request.args))
        imd = ImmutableMultiDict(request.args)
        mydict=imd.to_dict(flat=False)
        startdate = mydict['startdate']
        agent_list =mydict['agent']
        meal_list = mydict['meal']
        room_list = mydict['room']
        # print('TYPE=',type(mydict))
        # print('AGENT=',mydict['agent'])
       
        startdate = startdate[0]
        print(startdate)
        print("hiii",type(startdate))
        year = startdate[0:4]
        month = startdate[5:7]
        day =startdate[8:10]
        # print(mydict)
        mydate = dt.date(int(year),int(month),int(day))
        print(agent_list)
        print(meal_list)
        print(room_list)
        a,b, mygraphurls ,new_df, figurl=rl.getResults(agent_list,room_list,meal_list,mydate)
        graphkeys = mygraphurls.keys()
        graphkeyslen = len(graphkeys)
        # print(figurl)
        return flask.render_template('predictors.html',
                                     agents=agents,
                                     meals=meals,
                                     rooms=rooms,
                                     agents_len=agents_len,
                                     meals_len=meals_len,
                                     rooms_len=rooms_len,
                                     mygraphurls=mygraphurls,
                                     graphkeys=graphkeys,
                                     graphkeyslen=graphkeyslen,
                                     figurl=figurl,
                                     new_df=new_df,
                                     new_df_titles = new_df.columns.values,
                                     tables=dataset.head(10),
                                     titles=dataset.columns.values)
    else: 
        #For first load, request.args will be an empty ImmutableDict
        # type. If this is the case we need to pass an empty string
        # into make_prediction function so no errors are thrown.
        return retfun()
                
def retfun():
    return flask.render_template('predictors.html',
                                     agents=agents,
                                     meals=meals,
                                     rooms=rooms,
                                     agents_len=agents_len,
                                     meals_len=meals_len,
                                     rooms_len=rooms_len,
                                     tables=dataset.head(10), 
                                     titles=dataset.columns.values)



# # Start the server, continuously listen to requests.
if __name__=="__main__":
    # For local development, set to True:
    app.run(debug=False)
    # For public web serving:
    #app.run(host='0.0.0.0')
    app.run()