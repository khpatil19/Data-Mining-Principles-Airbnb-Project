import pandas as pd
import streamlit as st
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# import streamlit.components.v1 as components
# my_component = components.declare_component(
# "my_component",
# url="http://127.0.0.1:53878/form"
# )

# st.set_page_config(
#     page_title="Optimizer",
#     page_icon="👋",
# )

st.title("Airbnb Listing Optimizer")



with st.form("form1", clear_on_submit= False): 

    neighborhood_group_input = st.selectbox(
        'Where is your listing?',
        ('Brooklyn','Manhattan','Bronx','Queens','Staten Island'))

    st.write('You selected:', neighborhood_group_input)

    room_type_input = st.selectbox(
        'What type of room?',
        ('Entire home/apt', 'Private room', 'Shared room'))

    st.write('You selected:', room_type_input)

    # min 1, max 89

    # min_nights_input, max_nights_input = st.select_slider(
    #     'Select a range of nights',
    #     options = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
    # ,
    #     value=('1', '89'))
    
    # min_nights_input = st.number_input('Insert a number')
    # st.write('You selected ', min_nights_input)

    min_nights_input = st.slider('Enter minimum nights', 1, 90, 3)
    st.write('You selected ', str(min_nights_input))

    host_listings_input = st.slider('Enter number of existing listings', 1, 60, 10)
    st.write('You selected ', str(host_listings_input))

    submit = st.form_submit_button("Submit")

    if submit == True:
        #!/usr/bin/env python
        # coding: utf-8
        
        

        # In[1]:


    


        # In[2]:


        


        # ## Import Data

        # In[3]:


        data = pd.read_csv('AB_US_2023.csv')
        data = data.drop(['neighbourhood_group'],axis=1)
        data = data[data['city']=='New York City']


        # In[8]:


        # Numerical Variables\n",
        numerical_variables = ['price', 'minimum_nights','number_of_reviews', 'reviews_per_month',
                                'availability_365','number_of_reviews_ltm']

        # In[9]:


        # Filter out rows where 'price' is not null
        filtered_data = data.loc[data['price'].notnull(), 'price']


        numerical_df = data[numerical_variables]


        NYC_neighbourhood_mapping = pd.read_csv('NYC_neighbourhood_mapping.csv')

        # Rename the 'Region' column to 'neighbourhood_group'
        NYC_neighbourhood_mapping.rename(columns={'Region': 'neighbourhood_group'}, inplace=True)

        merged_data = pd.merge(data, NYC_neighbourhood_mapping, how='left', left_on='neighbourhood', right_on='Neighborhood')


        # In[13]:


        merged_data = merged_data.drop(['neighbourhood','Neighborhood'],axis=1)


        # In[14]:


        # List of categorical variables
        categorical_variables = ['neighbourhood_group','room_type']



        # In[15]:


        # Categorical Variables
        categorical_variables = ['neighbourhood_group','room_type']


        # ### Data Filtering

        # In[19]:


        #Check variation in price 
        merged_data[['neighbourhood_group','room_type','price']].groupby(['neighbourhood_group','room_type']).median('price')


        # In[216]:




        # In[217]:


        #Remove columns that are not required
        data_subset = merged_data.drop(['availability_365','latitude', 'longitude',
                                        'last_review','city','host_id', 'host_name','reviews_per_month'],axis=1)


        # In[218]:


        #Remove space from column names 
        data_subset.columns = data_subset.columns.str.replace(' ', '_')


        # In[219]:


        # Filter the DataFrame to include only rows where 'neighbourhood_group' is not null
        data_subset = data_subset[data_subset['neighbourhood_group'].notnull()]


        # In[220]:



        # In[226]:


        #Keep only data for private rooms and entire home/apt
        data_subset = data_subset[(data_subset['room_type'] == 'Entire home/apt') | (data_subset['room_type'] == 'Private room') | 
        (data_subset['room_type'] == 'Shared room')]


        # In[227]:



        # In[235]:



        # In[236]:


        #Remove outliers in price, >1262
        data_subset = data_subset[data_subset['price'] < 1262]


        # In[237]:


        # In[238]:


        #Remove outliers in minimum_nights, >90
        data_subset = data_subset[data_subset['minimum_nights'] < 90]


        # In[239]:

        # # Feature Engineering

        # In[241]:


        #Get dummies for categorical variables
        categorical_variables = ['neighbourhood_group','room_type']

        # Get dummy variables for categorical variables
        dummy_variables = pd.get_dummies(data_subset[categorical_variables], prefix=categorical_variables ) #, drop_first=True)

        # Concatenate the dummy variables with the original DataFrame
        data_subset = pd.concat([data_subset, dummy_variables], axis=1)

        # Drop the original categorical columns
        data_subset.drop(columns=categorical_variables, inplace=True)

        # Drop id column also
        #data_subset.drop('id',axis=1,inplace=True)


        # # Clustering

        # ### KMeans

        # In[242]:

        import sklearn
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans


        # In[246]:


        selected_features = ['minimum_nights', 'neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
                            'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island', 'calculated_host_listings_count',
                            'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room']

        model_data = data_subset[selected_features]


        # In[247]:



        # In[249]:


        #Step 1: Feature Scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(model_data)


        # In[250]:


        #Step 2: Dimensionality reduction

        # Create a PCA instance
        pca = PCA().fit(scaled_data)

        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)


        # In[251]:


        #Taking the first 6 PCs and transforming the data as they explain >80% of the variance
        reduced_data = pca.fit_transform(scaled_data)
        reduced_data = reduced_data[:,[0,1,2,3,4,5]]


        # In[252]:


        # Step 3: Choosing the Number of Clusters
        inertia = []
        x_axis = list(range(1,12))

        for x in x_axis:
            kmeans = KMeans(n_clusters=x)
            kmeans.fit(reduced_data)
            inertia.append(kmeans.inertia_)  



        # In[253]:


        # Step 4: K-means Clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(reduced_data)
        cluster_labels = kmeans.labels_


        # #### Visualise K-Means clustering output

        # In[254]:





        # In[188]:


        # Visualize Clusters Using Log-Transformed Features (K-means Model)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Rename the labels for clarity
        cluster_labels = kmeans.labels_



        # #### Assign cluster labels to data

        # In[255]:


        #Assign cluster labels to data
        data_subset['cluster_label'] = cluster_labels


        # In[256]:




        # #### Check average metrics for clusters

        # In[298]:




        # In[299]:




        # ## Clustering based price recommendation

        # ### Get user input

        # In[420]:


        input_neighbourhood_group = neighborhood_group_input
        input_room_type = room_type_input
        input_min_nights = min_nights_input
        input_host_listings = host_listings_input


        # In[421]:


        input_data = model_data.iloc[[0]].copy() * 0.0
        input_data['minimum_nights'] = input_min_nights
        input_data['calculated_host_listings_count'] = input_host_listings
    

        neighbourhood_groups = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        for neighbourhood_group in neighbourhood_groups:
            if neighbourhood_group == input_neighbourhood_group:
            
                var = 'neighbourhood_group_' + neighbourhood_group
                input_data[var] = True
            else:
                var = 'neighbourhood_group_' + neighbourhood_group
                input_data[var] = False


        room_types = ['Entire home/apt', 'Private room', 'Shared room']
        for room_type in room_types:
            if room_type == input_room_type:
                var = 'room_type_' + room_type
                input_data[var] = True
            else:
                var = 'room_type_' + room_type
                input_data[var] = False


        # ### Predict cluster for input listing

        # In[422]:
        # st.write(input_data)



        #Scale and reduce input
        input_data_scaled = scaler.transform(input_data)
        input_data_reduced = pca.transform(input_data_scaled)
        input_data_reduced = input_data_reduced[:,[0,1,2,3,4,5]]

        #Predict cluster assignments for input data
        input_data['Cluster'] = kmeans.predict(input_data_reduced)


        # In[423]:


        input_cluster = input_data['Cluster'][0]


        # In[424]:


        # st.write('Input listing belongs to cluster:',input_cluster)


        # ### Get the cluster's properties to recommend price for input listing

        # In[425]:


        #Get the median price for the assigned cluster
        median_price = data_subset[data_subset['cluster_label']==input_cluster].groupby('cluster_label')['price'].agg(['median'])
        median_price = median_price.iloc[0,0]
        # median_price




        # In[426]:


        median_price_lb = round(median_price * 0.875,2)
        median_price_ub = round(median_price * 1.125,2)

        # print('Suggested price range: $',median_price_lb,'to $',median_price_ub)

        st.write('Suggested price range for your selection: \$ ', str(median_price_lb),' to \$',str(median_price_ub))

        earning_lb = median_price_lb * min_nights_input
        earning_ub = median_price_ub * min_nights_input

        #On an average, airbnbs take 20% of the price as cleaning charges
        cleaning_fee = 0.2 
        earning_lb = earning_lb * (1+cleaning_fee)
        earning_ub = earning_ub * (1+cleaning_fee)

        #On an average, host pays 3% service fee to AirBnb for every booking
        service_fee = 0.03
        total_earnings_lb = round(earning_lb * (1-service_fee),2)
        total_earnings_ub = round(earning_ub * (1-service_fee),2)

        st.write('Total earnings range for host, per booking for ',str(min_nights_input),' nights: \$',str(total_earnings_lb),'to \$',str(total_earnings_ub))

        # # Send data to the frontend using named arguments.
        # return_value = my_component(name="Blackbeard", ship="Queen Anne's Revenge")

        # # `my_component`'s return value is the data returned from the frontend.
        # st.write("Value = ", return_value)

        #Cost calculation for customer
        #Calculate cost to customer
        service_fee_airbnb = 0.14
        cost_to_customer_lb = round(earning_lb * (1+service_fee_airbnb),2)
        cost_to_customer_ub = round(earning_ub * (1+service_fee_airbnb),2)

        st.write('Total cost paid by customer, per booking for ',str(min_nights_input),' nights: \$',str(cost_to_customer_lb),'to \$',str(cost_to_customer_ub))

        import nltk
        import re
        nltk.download('stopwords')

        def clean_text(text):
            # Remove numbers, symbols, and other non-alphabetic characters
            text = str(text)
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
            return cleaned_text


        # In[428]:


        data_subset['name'] = data_subset['name'].apply(clean_text)
        data_subset['name'] = data_subset['name'].to_list()


        # In[429]:


        #Customise stop words

        # Preprocessing stop words
        import nltk.corpus
        stopwords = set(nltk.corpus.stopwords.words('english'))

        stoplist = ['entire','private','queen','williamsburg','west','manhattan','bronx','upper','studio','nyc','brooklyn','queens','harlem',
                    'bushwick','astoria','midtown','central','east','village','room', 'bedroom', 'br', 'bdrm', 'bdr', 'house', 'apartment', 'apt',
                    'park','shared','hotel','days']

        for word in stopwords:
            stoplist.append(word)


        # In[430]:


        # Verctorize text
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(stop_words=stoplist)
        name = vectorizer.fit_transform(data_subset['name'])


        # In[431]:


        # Apply TF-IDF transformation
        from sklearn.feature_extraction.text import TfidfTransformer

        tfidf_transformer = TfidfTransformer()
        name_tfidf = tfidf_transformer.fit_transform(name)


        # In[432]:


        # data_subset.columns


        # In[458]:


        # Get a threshold for more reviewed listings from that cluster
        median_reviews = data_subset[data_subset['cluster_label']==input_cluster].groupby('cluster_label')['number_of_reviews'].agg(['median'])
        median_reviews_threshold = median_reviews.iloc[0,0]


        # In[434]:


        # Extract keywords based on TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        top_keywords = []

        # Filter the data for assigned cluster and for listings that cross the median review threshold 
        indices = data_subset.index[(data_subset['cluster_label'] == 1) & (data_subset['number_of_reviews'] == median_reviews_threshold)]    

        # Calculate TF-IDF scores
        valid_indices = [i for i in indices if i < name_tfidf.shape[0]]
        group_tfidf_scores = name_tfidf[valid_indices].mean(axis=0).A1

        # Get top keywords
        top_keyword_indices = group_tfidf_scores.argsort()[::-1][:20]  
        top_keywords = [feature_names[idx] for idx in top_keyword_indices]

        # Display recommended keywords 
        print('Recommended keywords for the given listing:', top_keywords)


        # In[435]:


        # Create a dictionary 
        dict_top_keywords = {}
        weight = 20
        for keyword in top_keywords:
            dict_top_keywords[keyword] = weight
            weight = weight - 1


        # In[436]:


        # Create a word cloud from top keywords
        # import graphviz
        from wordcloud import WordCloud

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict_top_keywords)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Plot the word cloud
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

        st.subheader('Recommended keywords for your listing name:')
        st.pyplot()





