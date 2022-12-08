import pandas as pd
import numpy as np
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
from PIL import Image

#streamlit setup begin
# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Content Based Recommended System", page_icon=":tada:", layout="wide")
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

#header
# st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# st.markdown("""
# <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #222222;">
#   <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Data Professor</a>
#   <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
#     <span class="navbar-toggler-icon"></span>
#   </button>
#   <div class="collapse navbar-collapse" id="navbarNav">
#     <ul class="nav navbar-nav navbar-right">
#       <li class="nav-item active">
#         <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
#       </li>
#       <li class="nav-item">
#         <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
#       </li>
#     </ul>
#   </div>
# </nav>
# """, unsafe_allow_html=True)

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
#img_contact_form = Image.open("images/yt_contact_form.png")
#img_lottie_animation = Image.open("images/yt_lottie_animation.png")

#loading bar for init
# my_bar = st.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)
# with st.spinner('Wait for it...'):

choice = st.sidebar.selectbox(
    'Navigation Bar',
    ('Home', 'Visualization of Data','About Us')
)

if choice == "Home":
    st.title("Content Based Smart Recommended System")
    my_bar = st.progress(0)
    with st.spinner('Loading All the Default Grind  ...'):
        progresss = 0
        #setup the system
        dataset = pd.read_csv("amazon_1.csv")
        dataset["Category"].head()
        my_bar.progress(20)
        cols = [0,2,3,5,6,8,9,12,14,16,17,18,19,20,21,22,23,24,25,26,27]
        dataset.drop(dataset.columns[cols], axis =1, inplace=True)
        dataset.dropna(inplace = True)
        dataset['Selling Price_processed'] = dataset['Selling Price'].apply(lambda x: str(x).replace('$',''))
        dataset['Selling Price_processed'] = pd.to_numeric(dataset['Selling Price_processed'], errors='coerce')
        dataset['Selling Price'] = dataset['Selling Price'].apply(lambda x: str(x).replace('$',''))
        dataset['Selling Price'] = pd.to_numeric(dataset['Selling Price'], errors='coerce')
        dataset = dataset.dropna()
        
        dataset.reset_index()
        my_bar.progress(50)
        # We can compute the similarity between categories using TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        dataset["Category"] = dataset["Category"].fillna("")
        tfidf_matrix = tfidf.fit_transform(dataset["Category"])
        my_bar.progress(55)
        # tfidf_matrix.shape
        # tfidf.get_feature_names_out()[0:20]
        from sklearn.metrics.pairwise import linear_kernel
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import sigmoid_kernel
        linear = linear_kernel(tfidf_matrix, tfidf_matrix)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sig_score = sigmoid_kernel(tfidf_matrix, tfidf_matrix)
        my_bar.progress(60)
        dataset = dataset.set_index([pd.Index(np.arange(0, 7045))])
        indices = pd.Series(dataset.index, index=dataset["Product Name"])
        my_bar.progress(progresss + 80)
        from fuzzywuzzy import fuzz
        from fuzzywuzzy import process
        choices = list(indices.index)
        extracted = process.extract("lego", choices, limit=1)
        # extracted[0][0]
        # Function that takes in product name as input and outputs most similar product

        # new data frame with split value columns. We use n = 3 to get a maximum of 3+1 columns
        new = dataset["Category"].str.split("|", n = 3, expand = True)
        
        dataset["Main Category"]= new[0] 
        
        dataset["Sub-Category"]= new[1]
        
        dataset["Side Category"]= new[2]

        dataset["Other Categories"]= new[3]
        
        dataset.drop(columns =["Category", "Other Categories"], inplace = True)
        my_bar.progress(90)
        #splitting of image column
        new=dataset["Image"].str.split("|",n=4,expand=True)

        dataset["1st image"]=new[0]
        dataset["2nd image"]=new[1]
        dataset["3rd image"]=new[2]
        dataset["4th image"]=new[3]
        dataset["5th image"]=new[4]
        my_bar.progress(100)
        dataset.drop(columns="Image",inplace=True)
        
    my_bar.empty()
    # st.balloons()
    # st.snow()


    # ---- HEADER SECTION ----

    # ---- WHAT I DO ----
    with st.container():
        st.subheader("A System Which identically recommend product by Special Smart Algorithm")
        st.write("Project Contain Demo Products from Amazon.com so search according.")
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("How this Smart System Work !!")
            st.write("##")
            st.write(
                """
                Search for products, brands and more
                
                """
            )
            #hyperlink adding format
            # st.write("[Channel >](https://channel.com)")
        with right_column:
            st_lottie(lottie_coding, height=300, key="coding")
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('System Loaded All The Data You can Begin....')

    def rec_lin(user_input,linear=linear):
        with st.spinner('Wait for it...'):
            extracted = process.extract(user_input, choices, limit=1)
            product_name = extracted[0][0]
            
            idx = indices[product_name]

            sim_scores = list(enumerate(linear[idx]))

            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            sim_scores = sim_scores[1:11]

            product_indices = [i[0] for i in sim_scores]

            df_return = dataset[["Product Name","Selling Price", "Selling Price_processed","Main Category","1st image"]].loc[product_indices]
            # Return the top 10 most similar products
            return df_return.sort_values(by="Selling Price_processed", ascending=True)[["Product Name","Selling Price","Main Category","1st image"]]
    

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    if "placeholder" not in st.session_state:
        st.session_state.placeholder = "visible"
        st.session_state.disabled = False

    st.markdown("<h3 style='text-align: center; color: black;'>What would you like to search for today?</h3>", unsafe_allow_html=True)
    # st.header("What would you like to search for today?")
    user_input = st.text_input(label = "Enter Name of Product to be search",label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder)
    if user_input:
        output_dataframe = rec_lin(user_input)
        st.write("---")
        st.dataframe(output_dataframe.set_index(["Product Name"]))
        for_footer_scroll = output_dataframe[["Product Name","Selling Price", "1st image"]]
        heads_for_dataset_graph = output_dataframe[["Product Name","Selling Price"]]
        # for i in heads_for_dataset_graph["Image"]:
        #     st.markdown("""<img src="https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885__340.jpg" 
        # alt="Cheetah!" />""".format(i), unsafe_allow_html=True)
        # eeee = output_dataframe["1st image"]
    else:
        output_dataframe = rec_lin(user_input)
        st.write("---")
        st.dataframe(output_dataframe.set_index(["Product Name"]))
        for_footer_scroll = output_dataframe[["Product Name","Selling Price", "1st image"]]
        heads_for_dataset_graph = output_dataframe[["Product Name","Selling Price"]]


    st.write("---")
    col0,col1,col2= st.columns(3)
    with col0:
        # st.text(for_footer_scroll["1st image"].iloc[0])
            #     st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[i]}" 
            # # alt="Cheetah!" />""".format(i), unsafe_allow_html=True)
        #         # st.markdown("{}".format(for_footer_scroll["1st image"].iloc[i]), unsafe_allow_html=True)
        # st.image(for_footer_scroll["1st image"].iloc[0],use_column_width=True,width=100,)
        # st.caption(for_footer_scroll["Product Name"].iloc[0])
        # st.markdown(str(int(for_footer_scroll["Selling Price"].iloc[0])*81.64) + " ₹")
                # st.text()
        # st.caption(for_footer_scroll["Product Name"].iloc[0])
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[0]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[0])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[0]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
        # st.image(for_footer_scroll["1st image"].iloc[0],use_column_width=True,width=100)
        #         # st.text()
    with col1:
        # st.caption(for_footer_scroll["Product Name"].iloc[1])
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[1]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[1])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[1]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[2]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[2])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[2]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
        st.text("")
    col3,col4,col5= st.columns(3)
    with col3:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[3]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[3])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[3]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
                    # st.text()
    with col4:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[4]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[4])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[4]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[5]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[5])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[5]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
        st.text("")
    col6,col7,col8= st.columns(3)
    with col6:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[6]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[6])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[6]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
                    # st.text()
    with col7:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[7]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[7])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[7]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    with col8:
        st.markdown(f"""<p class='imagetext'>{for_footer_scroll['Product Name'].iloc[8]}</p>""", unsafe_allow_html=True)
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[8])
        st.markdown(str(round(rupees_value,2)) + " $")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[8]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<h3 style='text-align: center; color: #4C5270;'>Visualization of Search Result</h3>", unsafe_allow_html=True)
        # st.head("Visualization of Search Result")
    dddd = heads_for_dataset_graph.set_index(["Selling Price"])
    st.line_chart(dddd)
    st.markdown("<h3 style='text-align: center; color: #4C5270;'>Line Graph of Product v/s Price</h3>", unsafe_allow_html=True)

if choice == "Visualization of Data":
    st.title("Insights from Recent Product Purchase & Visualization of Data page content")
    def visualization_fun():
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.2f}'.format)
        pd.set_option('display.max_colwidth', None)
        dataset = pd.read_csv("amazon_1.csv")
        dataset["Category"].head()
        cols = [0,2,3,5,6,8,9,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
        dataset.drop(dataset.columns[cols], axis =1, inplace=True)
        dataset.dropna(inplace = True)
        dataset['Selling Price_processed'] = dataset['Selling Price'].apply(lambda x: str(x).replace('$',''))
        dataset['Selling Price_processed'] = pd.to_numeric(dataset['Selling Price_processed'], errors='coerce')
        dataset = dataset.dropna()
        dataset.reset_index()
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        dataset["Category"] = dataset["Category"].fillna("")
        tfidf_matrix = tfidf.fit_transform(dataset["Category"])
        # tfidf_matrix.shape
        # tfidf.get_feature_names_out()[0:20]
        from sklearn.metrics.pairwise import linear_kernel
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import sigmoid_kernel
        linear = linear_kernel(tfidf_matrix, tfidf_matrix)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sig_score = sigmoid_kernel(tfidf_matrix, tfidf_matrix)
        df = dataset.copy()
        # new data frame with split value columns. We use n = 3 to get a maximum of 3+1 columns
        new = df["Category"].str.split("|", n = 3, expand = True)
        df["Main Category"]= new[0] 
        df["Sub-Category"]= new[1]
        df["Side Category"]= new[2]
        df["Other Categories"]= new[3]
        df.drop(columns =["Category", "Other Categories"], inplace = True)
        # df.head(2)
        df['Shipping Weight'] = df['Shipping Weight'].astype(str).str.strip('ounces')
        df['Shipping Weight'] = df['Shipping Weight'].str.strip('pounds')
        df['Shipping Weight'] = pd.to_numeric(df['Shipping Weight'], errors='coerce')
        df['Selling Price'] = df['Selling Price'].str.replace('$', '')
        df['Selling Price'] = pd.to_numeric(df['Selling Price'], errors='coerce')
        # df.head(2)
        df.dropna(inplace = True)
        df["Main Category"].unique()
        st.markdown("<p class='bordertextt';>Product count by category</p>", unsafe_allow_html=True)
        order = df['Main Category'].value_counts()[:10].index
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(y='Main Category', data=df, order=order)
        plt.title("Product count by category")
        plt.xlabel("Main category")
        plt.ylabel("Count of products")
        st.pyplot(fig)

        st.markdown("<p class='bordertextt';>Distribution of the Prices in Toys & Games Category</p>", unsafe_allow_html=True)
        fig2 = plt.figure(figsize=(10, 4))
        toys = df[df["Main Category"] == 'Toys & Games ']
        sns.boxplot(data = toys, x='Main Category', y='Selling Price', showfliers=False)
        plt.title("Distribution of the Prices in Toys & Games Category")
        st.pyplot(fig2)

        st.markdown("<p class='bordertextt';>Relationship between Price & Shipping Weight in Toys & Games category</p>", unsafe_allow_html=True)
        fig3 = plt.figure(figsize=(10, 4))
        sns.scatterplot(data=toys, x="Selling Price", y="Shipping Weight")
        plt.title("Relationship between Price & Shipping Weight in Toys & Games category")
        st.pyplot(fig3)

        st.markdown("<p class='bordertextt';>Distribution of the Prices in Home & Kitchen Category</p>", unsafe_allow_html=True)
        fig4 = plt.figure(figsize=(10, 4))
        home = df[df["Main Category"] == 'Home & Kitchen ']
        sns.boxplot(data = home, x='Main Category', y='Selling Price', showfliers=False)
        plt.title("Distribution of the Prices in Home & Kitchen Category")
        st.pyplot(fig4)

        st.markdown("<p class='bordertextt';>Relationship between Price & Shipping Weight in Home & Kitchen Category</p>", unsafe_allow_html=True)
        fig5 = plt.figure(figsize=(10, 4))
        sns.scatterplot(data=home, x="Selling Price", y="Shipping Weight")
        plt.title("Relationship between Price & Shipping Weight in Home & Kitchen Category")
        st.pyplot(fig5)

        st.markdown("<p class='bordertextt';>Distribution of the Prices in Sports & Outdoors Category</p>", unsafe_allow_html=True)
        fig6 = plt.figure(figsize=(10, 4))
        sports = df[df["Main Category"] == 'Sports & Outdoors ']
        sns.boxplot(data = sports, x='Main Category', y='Selling Price', showfliers=False)
        plt.title("Distribution of the Prices in Sports & Outdoors Category")
        st.pyplot(fig6)

        st.markdown("<p class='bordertextt';>Relationship between Price & Shipping Weight in Sports & Outdoors Category</p>", unsafe_allow_html=True)
        fig7 = plt.figure(figsize=(10, 4))
        sns.scatterplot(data=sports, x="Selling Price", y="Shipping Weight")
        plt.title("Relationship between Price & Shipping Weight in Sports & Outdoors Category")
        st.pyplot(fig7)

        st.markdown("<p class='bordertextt';>Distribution of the Prices in Clothing, Shoes & Jewelry Category</p>", unsafe_allow_html=True)
        fig8 = plt.figure(figsize=(10, 4))
        csj = df[df["Main Category"] == 'Clothing, Shoes & Jewelry ']
        sns.boxplot(data = csj, x='Main Category', y='Selling Price', showfliers=False)
        plt.title("Distribution of the Prices in Clothing, Shoes & Jewelry Category")
        st.pyplot(fig8)

        st.markdown("<p class='bordertextt';>Relationship between Price & Shipping Weight in Clothing, Shoes & Jewelry Category</p>", unsafe_allow_html=True)
        fig9 = plt.figure(figsize=(10, 4))
        sns.scatterplot(data=csj, x="Selling Price", y="Shipping Weight")
        plt.title("Relationship between Price & Shipping Weight in Clothing, Shoes & Jewelry Category")
        st.pyplot(fig9)
    visualization_fun()

if choice == "About Us":
    st.write("About Us Page")
    # ---- CONTACT ----
    with st.container():
        st.write("---")
        st.header("Any Suggestion for the Project Let Us Know !")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()
