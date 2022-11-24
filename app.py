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
# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

# ---- LOAD ASSETS ----
# lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
img_contact_form = Image.open("images/yt_contact_form.png")
img_lottie_animation = Image.open("images/yt_lottie_animation.png")

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
    with st.spinner('Loading All the Default Grind ...'):
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
                Smart system work as follows:
                - Point Number 1.
                - Point Number 2.
                - Point Number 3.
                - Point Number 4"

                Now more textttttt.
                """
            )
            #hyperlink adding format
            # st.write("[Channel >](https://channel.com)")
        with right_column:
            # st_lottie(lottie_coding, height=300, key="coding")
            st.text("Here may be json image will be display")
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('System Loaded All The Data You can Begin....')

    def rec_lin(linear=linear):
        st.markdown("<h3 style='text-align: center; color: black;'>What would you like to search for today?</h3>", unsafe_allow_html=True)
        # st.header("What would you like to search for today?")
        user_input = st.text_input("",placeholder="Enter Product to be search")
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

    output_dataframe = rec_lin()
    st.write("---")
    st.dataframe(output_dataframe.set_index(["Product Name"]))
    for_footer_scroll = output_dataframe[["Product Name","Selling Price", "1st image"]]
    heads_for_dataset_graph = output_dataframe[["Product Name","Selling Price"]]
    # for i in heads_for_dataset_graph["Image"]:
    #     st.markdown("""<img src="https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885__340.jpg" 
    # alt="Cheetah!" />""".format(i), unsafe_allow_html=True)
    # eeee = output_dataframe["1st image"]


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
        st.caption(for_footer_scroll["Product Name"].iloc[0])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[0])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[0]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
        # st.image(for_footer_scroll["1st image"].iloc[0],use_column_width=True,width=100)
        #         # st.text()
    with col1:
        st.caption(for_footer_scroll["Product Name"].iloc[1])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[1])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[1]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    with col2:
        st.caption(for_footer_scroll["Product Name"].iloc[2])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[2])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[2]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    st.text("")
    col3,col4,col5= st.columns(3)
    with col3:
        st.caption(for_footer_scroll["Product Name"].iloc[3])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[3])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[3]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
                # st.text()
    with col4:
        st.caption(for_footer_scroll["Product Name"].iloc[4])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[4])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[4]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    with col5:
        st.caption(for_footer_scroll["Product Name"].iloc[5])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[5])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[5]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    st.text("")
    col6,col7,col8= st.columns(3)
    with col6:
        st.caption(for_footer_scroll["Product Name"].iloc[6])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[6])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[6]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
                # st.text()
    with col7:
        st.caption(for_footer_scroll["Product Name"].iloc[7])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[7])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[7]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    with col8:
        st.caption(for_footer_scroll["Product Name"].iloc[8])
        rupees_value = int(for_footer_scroll["Selling Price"].iloc[8])*81.64
        st.markdown(str(round(rupees_value,2)) + "₹")
        st.markdown(f"""<img src="{for_footer_scroll["1st image"].iloc[8]}" alt="Cheetah" width="200" height="200"/>""", unsafe_allow_html=True)
    st.text("")
    st.markdown("<h1 style='text-align: center; color: Turquoise;'>Visualization of Search Result</h1>", unsafe_allow_html=True)
    # st.head("Visualization of Search Result")
    dddd = heads_for_dataset_graph.set_index(["Selling Price"])
    st.line_chart(dddd)

if choice == "Visualization of Data":
    st.write("Visualization of Data page content")
    st.dataframe(
        order = df['Main Category'].value_counts()[:10].index
        sns.countplot(y='Main Category', data=df, order=order)
        plt.title("Product count by category")
        plt.xlabel("Main category")
        plt.ylabel("Count of products"))

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
