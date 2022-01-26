from operator import index
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}

from flask import Flask, session

app = Flask(__name__, static_url_path="", static_folder="")
# model = pickle.load(open('modell.pkl', 'rb'))

app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('index.html', btn_css='btn btn-primary btn-block btn-large',csv_uploaded='Predict')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():

        # In[1]:


    import pandas as pd
    import matplotlib.pyplot as plt


    # In[2]:


    # get_ipython().run_line_magic('matplotlib', 'inline')


    # In[3]:


    dfile = pd.read_csv('2011-2012_Solar_home_electricity_data_v2.csv', skiprows=1,parse_dates=['date'], dayfirst=True,
                        na_filter=False, dtype={'Row Quality': str})


    # In[4]:


    d0, d1 = dfile.date.min(), dfile.date.max()
    d0, d1


    # In[8]:


    from pandas.tseries.offsets import Day
    index = pd.date_range(d0, d1 + Day(1), freq='30T', closed='left')
    index


    # In[9]:


    customers = sorted(dfile.Customer.unique())
    print(customers)


    # In[10]:


    channels = dfile['Consumption Category'].unique()
    channels


    # In[11]:


    channels = ['GC', 'GG', 'CL']


    # In[12]:


    columns = pd.MultiIndex.from_product(
        (customers, channels), names=['Customer', 'Channel'])
    columns


    # In[13]:


    cols_index = pd.MultiIndex(
        levels=[customers, channels],
        codes=[[],[]],
        names=['Customer', 'Channel'])


    # In[14]:


    df = pd.DataFrame(index=index, columns=cols_index)


    # In[15]:


    missing_records = []

    for c in customers:
        dcust = dfile[dfile.Customer == c]
        
        
        print(c, end=', ')
        
        for ch in channels:
            d_c_cust = dcust[dcust['Consumption Category'] == ch]
            ts = d_c_cust.iloc[:,5:-1].values.ravel()
            if len(ts) != len(index):
                missing_records.append((c,ch, len(ts)))
            else:
                df[c, ch] = ts


    # In[16]:


    nempty_CL = 0
    missingrecords_others = []
    for (c, ch, len_ts) in missing_records:
        if ch=='CL' and len_ts==0:
            nempty_CL += 1
        else:
            missingrecords_others.append((c, ch, len_ts))
    nempty_CL


    # In[17]:


    d_27_CL = dfile[(dfile.Customer == 27) & (dfile['Consumption Category'] == 'CL')]
    d_27_CL['date'].iloc[[0,-1]]


    # In[18]:


    a=[]
    for i in range(301):
        j=i
        print(j)
        if i!=0:
            if 'CL' in df[j]:
                a.append(j)
    len(a)   
        


    # In[21]:


    DF = df[1]
    for i in a:
        DF = DF + df[i]
        
    DF =  DF - df[1]
    DF
    DF_net =  DF
    DF


    # In[22]:


    #dj = gc - gg +cl
    DF_net =  DF['GC'] - DF['GG'] + DF['CL']


    # In[28]:


    import numpy
    # convert values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


    # In[29]:


    DF1 = df[1]['GC'] - df[1]['GG']  +df[1]['CL']


    # In[30]:


    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    DF1=scaler.fit_transform(np.array(DF1).reshape(-1,1))


    # In[31]:


    ##splitting dataset into train and test split
    training_size=int(len(DF1)*0.65)
    test_size=len(DF1)-training_size
    train_data,test_data=DF1[0:training_size,:],DF1[training_size:len(DF1),:1]


    # In[32]:


    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)


    # In[33]:


    from sklearn.linear_model import LinearRegression
    lin_model=LinearRegression()


    # In[34]:


    lin_model.fit(X_train,y_train)


    # In[36]:


    lin_pred=lin_model.predict(X_test)
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (11,6)
    plt.title('Net load forecast by linear for customer 1')
    plt.plot(lin_pred,label='Linear_Regression_Predictions')
    plt.plot(ytest,label='Actual Sales')
    plt.legend(loc="upper left")
    plt.savefig('Net load forecast by linear regression for customer 1.png', dpi=150);
    # plt.show()

    # return render_template('index.html', img = '/Net load forecast by linear regression for customer 1.png')
    return render_template('result.html', img = '/Net load forecast by linear regression for customer 1.png')

if __name__ == "__main__":
    app.run(debug=True)